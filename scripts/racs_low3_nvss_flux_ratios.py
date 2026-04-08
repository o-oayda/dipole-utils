from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

from dipoleutils.utils import CrossMatch, DataLoader
from scripts.paf_temperature_lookup import get_mean_paf_temperatures_for_mjd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MATCH_TABLE_PATH = DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_matches.ecsv"
DEFAULT_TILE_TABLE_PATH = DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_tiles.ecsv"
DEFAULT_RATIO_PAYLOAD_PATH = DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_by_sbid.npy"
LOW3_OBSERVING_FREQUENCY_MHZ = 943.5
NVSS_OBSERVING_FREQUENCY_MHZ = 1400.0
DEFAULT_SPECTRAL_INDEX = 0.8
DEFAULT_MATCH_RADIUS_ARCSEC = 5.0


def scale_flux_density(
    flux_density,
    input_frequency_mhz: float = NVSS_OBSERVING_FREQUENCY_MHZ,
    target_frequency_mhz: float = LOW3_OBSERVING_FREQUENCY_MHZ,
    spectral_index: float = DEFAULT_SPECTRAL_INDEX,
):
    flux_density = np.asarray(flux_density, dtype=float)
    return flux_density * (target_frequency_mhz / input_frequency_mhz) ** (-spectral_index)


def load_racs_low3_catalogue() -> Table:
    return DataLoader("racs", "low3").load(
        columns=["Name", "RA", "Dec", "Total_flux", "SBID", "Scan_start_MJD"]
    )


def load_nvss_catalogue() -> Table:
    return DataLoader("nvss").load(
        columns=["source_name", "ra", "dec", "integrated_flux"]
    )


def build_crossmatched_flux_ratio_products(
    racs_catalogue: Table,
    nvss_catalogue: Table,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    spectral_index: float = DEFAULT_SPECTRAL_INDEX,
) -> tuple[Table, Table, dict[str, object]]:
    crossmatch = CrossMatch(
        racs_catalogue,
        nvss_catalogue,
        coordinate_system="equatorial",
    )
    crossmatch.cross_match(
        radius=match_radius_arcsec,
        source_name_A_column="Name",
        source_name_B_column="source_name",
    )
    matched = crossmatch.get_common_sources()

    racs_flux = np.asarray(matched["A_Total_flux"], dtype=float)
    nvss_flux = np.asarray(matched["B_integrated_flux"], dtype=float)
    scaled_nvss_flux = scale_flux_density(nvss_flux, spectral_index=spectral_index)
    valid_flux = (
        np.isfinite(racs_flux)
        & np.isfinite(nvss_flux)
        & np.isfinite(scaled_nvss_flux)
        & (scaled_nvss_flux > 0.0)
    )
    matched = matched[valid_flux]

    racs_flux = np.asarray(matched["A_Total_flux"], dtype=float)
    nvss_flux = np.asarray(matched["B_integrated_flux"], dtype=float)
    scaled_nvss_flux = scale_flux_density(nvss_flux, spectral_index=spectral_index)
    flux_ratio = racs_flux / scaled_nvss_flux
    sbid = np.asarray(matched["A_SBID"], dtype=np.int64)
    scan_start_mjd = np.asarray(matched["A_Scan_start_MJD"], dtype=float)

    unique_sbid, first_indices, counts = np.unique(
        sbid,
        return_index=True,
        return_counts=True,
    )
    tile_scan_start_mjd = scan_start_mjd[first_indices]
    tile_paf_temperature_c = get_mean_paf_temperatures_for_mjd(tile_scan_start_mjd)

    sbid_to_temperature = {
        int(tile_sbid): float(tile_temperature)
        for tile_sbid, tile_temperature in zip(unique_sbid, tile_paf_temperature_c, strict=True)
    }
    matched["Scaled_NVSS_flux_943p5MHz"] = scaled_nvss_flux
    matched["Flux_ratio_RACS_over_NVSS_scaled"] = flux_ratio
    matched["Mean_PAF_Temperature_C"] = np.asarray(
        [sbid_to_temperature[int(tile_sbid)] for tile_sbid in sbid],
        dtype=float,
    )

    ratio_arrays_by_sbid: dict[int, np.ndarray] = {}
    tile_mean_ratio = np.empty(unique_sbid.size, dtype=float)
    tile_median_ratio = np.empty(unique_sbid.size, dtype=float)
    tile_std_ratio = np.empty(unique_sbid.size, dtype=float)
    for tile_index, tile_sbid in enumerate(unique_sbid):
        tile_ratios = np.asarray(flux_ratio[sbid == tile_sbid], dtype=float)
        ratio_arrays_by_sbid[int(tile_sbid)] = tile_ratios
        tile_mean_ratio[tile_index] = float(np.mean(tile_ratios))
        tile_median_ratio[tile_index] = float(np.median(tile_ratios))
        tile_std_ratio[tile_index] = float(np.std(tile_ratios, dtype=float))

    tile_table = Table(
        {
            "SBID": unique_sbid.astype(np.int64),
            "Scan_start_MJD": tile_scan_start_mjd,
            "Mean_PAF_Temperature_C": tile_paf_temperature_c,
            "N_Matches": counts.astype(np.int64),
            "Mean_Flux_ratio_RACS_over_NVSS_scaled": tile_mean_ratio,
            "Median_Flux_ratio_RACS_over_NVSS_scaled": tile_median_ratio,
            "Std_Flux_ratio_RACS_over_NVSS_scaled": tile_std_ratio,
        }
    )

    ratio_payload: dict[str, object] = {
        "match_radius_arcsec": float(match_radius_arcsec),
        "spectral_index": float(spectral_index),
        "nvss_observing_frequency_mhz": float(NVSS_OBSERVING_FREQUENCY_MHZ),
        "racs_low3_observing_frequency_mhz": float(LOW3_OBSERVING_FREQUENCY_MHZ),
        "tile_summary": {
            "SBID": unique_sbid.astype(np.int64),
            "Scan_start_MJD": tile_scan_start_mjd,
            "Mean_PAF_Temperature_C": tile_paf_temperature_c,
            "N_Matches": counts.astype(np.int64),
            "Mean_Flux_ratio_RACS_over_NVSS_scaled": tile_mean_ratio,
            "Median_Flux_ratio_RACS_over_NVSS_scaled": tile_median_ratio,
            "Std_Flux_ratio_RACS_over_NVSS_scaled": tile_std_ratio,
        },
        "ratio_arrays_by_sbid": ratio_arrays_by_sbid,
    }
    return matched, tile_table, ratio_payload


def plot_flux_ratio_vs_temperature(
    matched_table: Table,
    tile_table: Table,
) -> tuple[plt.Figure, plt.Axes]:
    point_temperature_c = np.asarray(matched_table["Mean_PAF_Temperature_C"], dtype=float)
    point_flux_ratio = np.asarray(
        matched_table["Flux_ratio_RACS_over_NVSS_scaled"],
        dtype=float,
    )
    tile_temperature_c = np.asarray(tile_table["Mean_PAF_Temperature_C"], dtype=float)
    tile_mean_ratio = np.asarray(
        tile_table["Mean_Flux_ratio_RACS_over_NVSS_scaled"],
        dtype=float,
    )
    tile_std_ratio = np.asarray(
        tile_table["Std_Flux_ratio_RACS_over_NVSS_scaled"],
        dtype=float,
    )

    valid_points = np.isfinite(point_temperature_c) & np.isfinite(point_flux_ratio)
    valid_tiles = (
        np.isfinite(tile_temperature_c)
        & np.isfinite(tile_mean_ratio)
        & np.isfinite(tile_std_ratio)
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        point_temperature_c[valid_points],
        point_flux_ratio[valid_points],
        s=8,
        color="black",
        alpha=1.0,
        linewidths=0,
    )
    axis.errorbar(
        tile_temperature_c[valid_tiles],
        tile_mean_ratio[valid_tiles],
        yerr=tile_std_ratio[valid_tiles],
        fmt="o",
        color="tab:red",
        ecolor="tab:red",
        elinewidth=1.2,
        capsize=0,
        markersize=4,
    )
    axis.set_xlabel("Mean PAF Temperature Per Tile (C)")
    axis.set_ylabel("Flux Ratio RACS / NVSS Scaled to 943.5 MHz")
    axis.set_title("LOW3 Tile Flux-Ratio Distribution vs PAF Temperature")
    figure.tight_layout()
    return figure, axis





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-radius-arcsec",
        type=float,
        default=DEFAULT_MATCH_RADIUS_ARCSEC,
    )
    parser.add_argument(
        "--spectral-index",
        type=float,
        default=DEFAULT_SPECTRAL_INDEX,
    )
    parser.add_argument(
        "--match-table-path",
        type=Path,
        default=DEFAULT_MATCH_TABLE_PATH,
    )
    parser.add_argument(
        "--tile-table-path",
        type=Path,
        default=DEFAULT_TILE_TABLE_PATH,
    )
    parser.add_argument(
        "--ratio-payload-path",
        type=Path,
        default=DEFAULT_RATIO_PAYLOAD_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    racs_catalogue = load_racs_low3_catalogue()
    nvss_catalogue = load_nvss_catalogue()
    matched_table, tile_table, ratio_payload = build_crossmatched_flux_ratio_products(
        racs_catalogue,
        nvss_catalogue,
        match_radius_arcsec=args.match_radius_arcsec,
        spectral_index=args.spectral_index,
    )

    finite_tile_temperatures = np.isfinite(
        np.asarray(tile_table["Mean_PAF_Temperature_C"], dtype=float)
    )
    print(f"LOW3 rows loaded: {len(racs_catalogue)}")
    print(f"NVSS rows loaded: {len(nvss_catalogue)}")
    print(f"Valid crossmatches saved: {len(matched_table)}")
    print(f"SBID tiles with matches: {len(tile_table)}")
    print(f"Tiles with finite mean PAF temperature: {int(np.sum(finite_tile_temperatures))}")
    plot_flux_ratio_vs_temperature(matched_table, tile_table)
    plt.show()
