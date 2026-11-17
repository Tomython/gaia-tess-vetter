import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import lightkurve as lk

warnings.filterwarnings("ignore")

class GaiaTessVetter:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)

    def query_gaia_and_neighbors(self, source_id, search_radius_arcsec=21.0):
        """Запрашивает целевую звезду в Gaia DR3 и проверяет соседей быстрым синхронным ADQL-запросом."""
        query = f"""
        SELECT source_id, ra, dec, parallax, pmra, pmdec,
               phot_g_mean_mag, bp_rp, ruwe, teff_gspphot, logg_gspphot
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
        """
        job = Gaia.launch_job(query)
        target_tbl = job.get_results()
        if len(target_tbl) == 0:
            return None, None

        target = target_tbl[0]
        ra_val = float(target['ra'])
        dec_val = float(target['dec'])

        # Быстрый синхронный поиск соседей в квадрате вокруг звезды (рекомендация ЕКА вместо cone search)
        deg_rad = search_radius_arcsec / 3600.0
        cos_dec = max(np.cos(np.radians(dec_val)), 0.01)
        ra_min, ra_max = ra_val - deg_rad / cos_dec, ra_val + deg_rad / cos_dec
        dec_min, dec_max = dec_val - deg_rad, dec_val + deg_rad

        neighbor_query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE ra BETWEEN {ra_min} AND {ra_max}
          AND dec BETWEEN {dec_min} AND {dec_max}
        """
        n_job = Gaia.launch_job(neighbor_query)
        neighbors = n_job.get_results()
        return target, neighbors

    def analyze_tess_lightcurve(self, coord):
        """Скачивает кривые блеска TESS, чистит их и запускает поиск транзитов BLS."""
        search_result = lk.search_lightcurve(coord, mission="TESS", author="SPOC")
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(coord, mission="TESS")
        
        if len(search_result) == 0:
            return None

        # Скачиваем до 4 последних секторов
        lc_collection = search_result[-4:].download_all()
        if lc_collection is None:
            return None

        lc = lc_collection.stitch().remove_nans().remove_outliers(sigma=5)
        flat_lc = lc.flatten(window_length=401)

        # Запускаем Box Least Squares (BLS) периодограмму
        periodogram = flat_lc.to_periodogram(method="bls", period=np.linspace(0.5, 25.0, 5000))
        best_period = float(periodogram.period_at_max_power.value)
        best_t0 = float(periodogram.transit_time_at_max_power.value)
        best_dur = float(periodogram.duration_at_max_power.value)
        max_power = float(periodogram.max_power)

        # Сворачиваем кривую блеска по найденному периоду
        folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)
        
        # Оценка глубины транзита и проверка на вторичное затмение
        phase_days = folded_lc.phase.value
        in_transit = np.abs(phase_days) < (best_dur / 2.0)
        sec_eclipse = np.abs(np.abs(phase_days) - 0.5 * best_period) < (best_dur / 2.0)
        
        transit_depth = float(1.0 - np.nanmedian(folded_lc.flux.value[in_transit])) if np.any(in_transit) else 0.0
        sec_depth = float(1.0 - np.nanmedian(folded_lc.flux.value[sec_eclipse])) if np.any(sec_eclipse) else 0.0

        return {
            "sectors": len(lc_collection),
            "flat_lc": flat_lc,
            "periodogram": periodogram,
            "folded_lc": folded_lc,
            "best_period": best_period,
            "best_t0": best_t0,
            "best_dur_hours": best_dur * 24.0,
            "bls_power": max_power,
            "transit_depth_ppm": transit_depth * 1e6,
            "sec_depth_ppm": sec_depth * 1e6
        }

    def vet_target(self, source_id):
        print(f"[*] Проверка Gaia ID: {source_id}...")
        target, neighbors = self.query_gaia_and_neighbors(source_id)
        if target is None:
            print(f"[!] Звезда {source_id} не найдена в Gaia.")
            return None

        ruwe = float(target['ruwe']) if not np.ma.is_masked(target['ruwe']) else np.nan
        g_mag = float(target['phot_g_mean_mag'])
        coord = SkyCoord(ra=float(target['ra'])*u.deg, dec=float(target['dec'])*u.deg, frame='icrs')

        # Расчет загрязнения от соседей в пикселе TESS
        bright_neighbors = 0
        contamination_ratio = 0.0
        target_flux = 10**(-0.4 * g_mag)
        total_flux = target_flux

        if neighbors is not None and len(neighbors) > 1:
            for row in neighbors:
                if int(row['source_id']) != int(source_id) and not np.ma.is_masked(row['phot_g_mean_mag']):
                    n_mag = float(row['phot_g_mean_mag'])
                    total_flux += 10**(-0.4 * n_mag)
                    if n_mag < (g_mag + 2.5):
                        bright_neighbors += 1
            contamination_ratio = (total_flux - target_flux) / total_flux

        # Анализ TESS
        tess_data = self.analyze_tess_lightcurve(coord)

        # Формирование автоматического научного вердикта
        flags = []
        if not np.isnan(ruwe) and ruwe > 1.4:
            flags.append("HIGH_RUWE_BINARY_RISK")
        if contamination_ratio > 0.25:
            flags.append("CROWDED_TESS_PIXEL")
        if tess_data is None:
            flags.append("NO_TESS_DATA")
        else:
            if tess_data["sec_depth_ppm"] > 0.4 * tess_data["transit_depth_ppm"] and tess_data["transit_depth_ppm"] > 500:
                flags.append("SECONDARY_ECLIPSE_EB")
            if tess_data["transit_depth_ppm"] > 30000:
                flags.append("DEEP_ECLIPSE_STELLAR")

        verdict = "CLEAN_CANDIDATE" if len(flags) == 0 else " | ".join(flags)

        report = {
            "gaia_source_id": source_id,
            "ra": round(float(target['ra']), 6),
            "dec": round(float(target['dec']), 6),
            "g_mag": round(g_mag, 3),
            "ruwe": round(ruwe, 3) if not np.isnan(ruwe) else None,
            "teff_k": round(float(target['teff_gspphot']), 0) if not np.ma.is_masked(target['teff_gspphot']) else None,
            "bright_neighbors_21arcsec": bright_neighbors,
            "flux_contamination": round(contamination_ratio, 3),
            "tess_sectors": tess_data["sectors"] if tess_data else 0,
            "bls_period_days": round(tess_data["best_period"], 5) if tess_data else None,
            "transit_depth_ppm": round(tess_data["transit_depth_ppm"], 1) if tess_data else None,
            "bls_power": round(tess_data["bls_power"], 2) if tess_data else None,
            "verdict": verdict
        }

        if tess_data is not None:
            self.save_dashboard(source_id, report, neighbors, tess_data)

        return report

    def save_dashboard(self, source_id, report, neighbors, tess_data):
        """Генерирует карточку-дашборд для загрузки на ExoFOP."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Gaia ID {source_id} | G={report['g_mag']} | RUWE={report['ruwe']} | Verdict: {report['verdict']}", fontsize=12, fontweight='bold')

        # 1. Карта окружения в радиусе пикселя TESS (21'')
        ax1 = axes[0, 0]
        circle = plt.Circle((report['ra'], report['dec']), 21.0/3600.0, color='red', fill=False, linestyle='--', label='TESS Pixel (21")')
        ax1.add_patch(circle)
        if neighbors is not None:
            ax1.scatter(neighbors['ra'], neighbors['dec'], s=40, c='gray', alpha=0.7, label='Gaia Neighbors')
        ax1.scatter([report['ra']], [report['dec']], s=120, c='gold', edgecolors='black', marker='*', label='Target')
        ax1.set_title(f"Field Crowding (Contamination: {report['flux_contamination']*100:.1f}%)")
        ax1.set_xlabel("RA (deg)")
        ax1.set_ylabel("Dec (deg)")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Очищенная кривая блеска TESS
        ax2 = axes[0, 1]
        tess_data["flat_lc"].scatter(ax=ax2, s=1, c='black', alpha=0.4)
        ax2.set_title(f"TESS Flattened Light Curve ({tess_data['sectors']} sectors)")

        # 3. Периодограмма BLS
        ax3 = axes[1, 0]
        tess_data["periodogram"].plot(ax=ax3, c='navy')
        ax3.axvline(tess_data["best_period"], color='red', linestyle='--', alpha=0.7, label=f"Best P={report['bls_period_days']} d")
        ax3.set_title("BLS Periodogram")
        ax3.legend()

        # 4. Фазовая кривая блеска (транзит)
        ax4 = axes[1, 1]
        folded = tess_data["folded_lc"]
        folded.scatter(ax=ax4, s=2, c='gray', alpha=0.3)
        folded.bin(time_bin_size=0.02).scatter(ax=ax4, s=25, c='crimson', label='Binned Flux')
        ax4.set_title(f"Phase-Folded Transit (Depth: {report['transit_depth_ppm']} ppm)")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/gaia_{source_id}_vetting.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    vetter = GaiaTessVetter()
    test_ids = [
        4955371367334610048,  # WASP-18
        2826537805196550272   # HD 209458
    ]
    results = []
    for sid in test_ids:
        res = vetter.vet_target(sid)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("results/reports/test_vetting_report.csv", index=False)
    print("\n--- ИТОГОВЫЙ ОТЧЕТ ---")
    print(df.to_string())
