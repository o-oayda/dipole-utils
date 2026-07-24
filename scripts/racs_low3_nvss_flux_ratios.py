from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

from dipoleutils.utils import CrossMatch, DataLoader, CatalogueToMap
from dipoleutils import RACS
from scripts.paf_temperature_lookup import get_mean_paf_temperatures_for_mjd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MATCH_TABLE_PATH = DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_matches.ecsv"
DEFAULT_TEMPERATURE_BIN_TABLE_PATH = (
    DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_temperature_bins.ecsv"
)
DEFAULT_RATIO_PAYLOAD_PATH = DEFAULT_OUTPUT_DIR / "racs_low3_nvss_flux_ratio_by_sbid.npy"
LOW3_OBSERVING_FREQUENCY_MHZ = 943.5
NVSS_OBSERVING_FREQUENCY_MHZ = 1400.0
DEFAULT_SPECTRAL_INDEX = 0.8
DEFAULT_MATCH_RADIUS_ARCSEC = 5.0
DEFAULT_TEMPERATURE_BIN_WIDTH_C = 0.25
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0


def scale_flux_density(
    flux_density,
    input_frequency_mhz: float = NVSS_OBSERVING_FREQUENCY_MHZ,
    target_frequency_mhz: float = LOW3_OBSERVING_FREQUENCY_MHZ,
    spectral_index: float = DEFAULT_SPECTRAL_INDEX,
):
    flux_density = np.asarray(flux_density, dtype=float)
    return flux_density * (target_frequency_mhz / input_frequency_mhz) ** (-spectral_index)


def load_racs_low3_catalogue() -> Table:
    cat = DataLoader("racs", "low3").load(
        columns=["Name", "RA", "Dec", "Total_flux", "SBID", "Scan_start_MJD"]
    )
    sample = CatalogueToMap(cat)
    sample.make_cut('Total_flux', 15, None)
    return sample.get_catalogue()

def load_mid1_catalogue() -> Table:
    cat = RACS('mid1')
    columns = ["Name", "RA", "Dec", "Total_flux", "SBID", "Scan_start_MJD"]
    print(min(cat['Total_flux']))
    return cat[columns]

def load_nvss_catalogue() -> Table:
    cat = DataLoader("nvss").load(
        columns=["source_name", "ra", "dec", "integrated_flux"]
    )
    sample = CatalogueToMap(cat)
    sample.make_cut('integrated_flux', 15, 1000)
    return sample.get_catalogue()


def _bootstrap_mean_flux_ratio_by_tile(
    ratio_arrays_by_sbid: dict[int, np.ndarray],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> float:
    tile_ids = np.asarray(sorted(ratio_arrays_by_sbid), dtype=np.int64)
    if tile_ids.size <= 1:
        return 0.0

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(int(bootstrap_resamples), dtype=float)
    for bootstrap_index in range(int(bootstrap_resamples)):
        sampled_tile_ids = rng.choice(tile_ids, size=tile_ids.size, replace=True)
        sampled_ratios = np.concatenate(
            [ratio_arrays_by_sbid[int(tile_id)] for tile_id in sampled_tile_ids]
        )
        bootstrap_means[bootstrap_index] = float(np.mean(sampled_ratios))
    return float(np.std(bootstrap_means, ddof=1))


def _build_temperature_bin_summary(
    flux_ratio: np.ndarray,
    sbid: np.ndarray,
    temperature_c: np.ndarray,
    bin_width_c: float = DEFAULT_TEMPERATURE_BIN_WIDTH_C,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[Table, dict[float, np.ndarray]]:
    valid = np.isfinite(flux_ratio) & np.isfinite(temperature_c)
    flux_ratio = np.asarray(flux_ratio[valid], dtype=float)
    sbid = np.asarray(sbid[valid], dtype=np.int64)
    temperature_c = np.asarray(temperature_c[valid], dtype=float)

    if flux_ratio.size == 0:
        empty_table = Table(
            {
                "Temperature_bin_start_C": np.asarray([], dtype=float),
                "Temperature_bin_end_C": np.asarray([], dtype=float),
                "Temperature_bin_center_C": np.asarray([], dtype=float),
                "N_Tiles": np.asarray([], dtype=np.int64),
                "N_Matches": np.asarray([], dtype=np.int64),
                "Mean_Flux_ratio_RACS_over_NVSS_scaled": np.asarray([], dtype=float),
                "Median_Flux_ratio_RACS_over_NVSS_scaled": np.asarray([], dtype=float),
                "Std_Flux_ratio_RACS_over_NVSS_scaled": np.asarray([], dtype=float),
                "Bootstrap_uncertainty_on_mean_Flux_ratio": np.asarray([], dtype=float),
            }
        )
        return empty_table, {}

    temperature_min_c = float(np.min(temperature_c))
    bin_start_c = temperature_min_c + np.floor(
        (temperature_c - temperature_min_c) / bin_width_c
    ) * bin_width_c
    unique_bin_start_c = np.unique(bin_start_c)

    ratio_arrays_by_bin_start: dict[float, np.ndarray] = {}
    n_tiles = np.empty(unique_bin_start_c.size, dtype=np.int64)
    n_matches = np.empty(unique_bin_start_c.size, dtype=np.int64)
    mean_ratio = np.empty(unique_bin_start_c.size, dtype=float)
    median_ratio = np.empty(unique_bin_start_c.size, dtype=float)
    std_ratio = np.empty(unique_bin_start_c.size, dtype=float)
    bootstrap_uncertainty = np.empty(unique_bin_start_c.size, dtype=float)

    for bin_index, current_bin_start_c in enumerate(unique_bin_start_c):
        in_bin = bin_start_c == current_bin_start_c
        bin_ratios = np.asarray(flux_ratio[in_bin], dtype=float)
        bin_sbid = np.asarray(sbid[in_bin], dtype=np.int64)
        ratio_arrays_by_bin_start[float(current_bin_start_c)] = bin_ratios

        unique_tile_ids = np.unique(bin_sbid)
        ratio_arrays_by_sbid = {
            int(tile_id): np.asarray(bin_ratios[bin_sbid == tile_id], dtype=float)
            for tile_id in unique_tile_ids
        }

        n_tiles[bin_index] = unique_tile_ids.size
        n_matches[bin_index] = bin_ratios.size
        mean_ratio[bin_index] = float(np.mean(bin_ratios))
        median_ratio[bin_index] = float(np.median(bin_ratios))
        std_ratio[bin_index] = float(np.std(bin_ratios, dtype=float))
        bootstrap_uncertainty[bin_index] = _bootstrap_mean_flux_ratio_by_tile(
            ratio_arrays_by_sbid,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + bin_index,
        )

    temperature_bin_table = Table(
        {
            "Temperature_bin_start_C": unique_bin_start_c,
            "Temperature_bin_end_C": unique_bin_start_c + bin_width_c,
            "Temperature_bin_center_C": unique_bin_start_c + 0.5 * bin_width_c,
            "N_Tiles": n_tiles,
            "N_Matches": n_matches,
            "Mean_Flux_ratio_RACS_over_NVSS_scaled": mean_ratio,
            "Median_Flux_ratio_RACS_over_NVSS_scaled": median_ratio,
            "Std_Flux_ratio_RACS_over_NVSS_scaled": std_ratio,
            "Bootstrap_uncertainty_on_mean_Flux_ratio": bootstrap_uncertainty,
        }
    )
    return temperature_bin_table, ratio_arrays_by_bin_start


def build_crossmatched_flux_ratio_products(
    racs_catalogue: Table,
    nvss_catalogue: Table,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    spectral_index: float = DEFAULT_SPECTRAL_INDEX,
    temperature_bin_width_c: float = DEFAULT_TEMPERATURE_BIN_WIDTH_C,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
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

    ratio_arrays_by_sbid: dict[int, np.ndarray] = {
        int(tile_sbid): np.asarray(flux_ratio[sbid == tile_sbid], dtype=float)
        for tile_sbid in unique_sbid
    }
    temperature_bin_table, ratio_arrays_by_temperature_bin = _build_temperature_bin_summary(
        flux_ratio,
        sbid,
        np.asarray(matched["Mean_PAF_Temperature_C"], dtype=float),
        bin_width_c=temperature_bin_width_c,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    ratio_payload: dict[str, object] = {
        "match_radius_arcsec": float(match_radius_arcsec),
        "spectral_index": float(spectral_index),
        "temperature_bin_width_c": float(temperature_bin_width_c),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "nvss_observing_frequency_mhz": float(NVSS_OBSERVING_FREQUENCY_MHZ),
        "racs_low3_observing_frequency_mhz": float(LOW3_OBSERVING_FREQUENCY_MHZ),
        "tile_summary": {
            "SBID": unique_sbid.astype(np.int64),
            "Scan_start_MJD": tile_scan_start_mjd,
            "Mean_PAF_Temperature_C": tile_paf_temperature_c,
            "N_Matches": counts.astype(np.int64),
        },
        "temperature_bin_summary": {
            key: np.asarray(value)
            for key, value in temperature_bin_table.items()
        },
        "ratio_arrays_by_sbid": ratio_arrays_by_sbid,
        "ratio_arrays_by_temperature_bin": ratio_arrays_by_temperature_bin,
    }
    return matched, temperature_bin_table, ratio_payload


def plot_flux_ratio_vs_temperature(
    matched_table: Table,
    temperature_bin_table: Table,
) -> tuple[plt.Figure, plt.Axes]:
    point_temperature_c = np.asarray(matched_table["Mean_PAF_Temperature_C"], dtype=float)
    point_flux_ratio = np.asarray(
        matched_table["Flux_ratio_RACS_over_NVSS_scaled"],
        dtype=float,
    )
    bin_temperature_c = np.asarray(
        temperature_bin_table["Temperature_bin_center_C"],
        dtype=float,
    )
    bin_mean_ratio = np.asarray(
        temperature_bin_table["Mean_Flux_ratio_RACS_over_NVSS_scaled"],
        dtype=float,
    )
    bin_mean_uncertainty = np.asarray(
        temperature_bin_table["Bootstrap_uncertainty_on_mean_Flux_ratio"],
        dtype=float,
    )

    valid_points = np.isfinite(point_temperature_c) & np.isfinite(point_flux_ratio)
    valid_bins = (
        np.isfinite(bin_temperature_c)
        & np.isfinite(bin_mean_ratio)
        & np.isfinite(bin_mean_uncertainty)
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        point_temperature_c[valid_points],
        point_flux_ratio[valid_points],
        s=1,
        alpha=0.1,
        linewidths=0,
    )
    axis.errorbar(
        bin_temperature_c[valid_bins],
        bin_mean_ratio[valid_bins],
        yerr=bin_mean_uncertainty[valid_bins],
        fmt="o",
        color="tab:orange",
        ecolor="tab:orange",
        elinewidth=1.2,
        capsize=0,
        markersize=4,
    )
    axis.set_ylim(0.5, 1.5)
    axis.set_xlabel("Mean PAF Temperature Bin Centre (C)")
    axis.set_ylabel("Flux Ratio RACS / NVSS Scaled to 943.5 MHz")
    axis.set_title("LOW3 Flux-Ratio Distribution vs PAF Temperature")
    axis.axhline(y=1, zorder=10, linestyle='--', color='black')
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
        "--temperature-bin-table-path",
        type=Path,
        default=DEFAULT_TEMPERATURE_BIN_TABLE_PATH,
    )
    parser.add_argument(
        "--ratio-payload-path",
        type=Path,
        default=DEFAULT_RATIO_PAYLOAD_PATH,
    )
    parser.add_argument(
        "--temperature-bin-width-c",
        type=float,
        default=DEFAULT_TEMPERATURE_BIN_WIDTH_C,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    racs_catalogue = load_racs_low3_catalogue()
    # racs_catalogue = load_mid1_catalogue()
    nvss_catalogue = load_nvss_catalogue()
    matched_table, temperature_bin_table, ratio_payload = build_crossmatched_flux_ratio_products(
        racs_catalogue,
        nvss_catalogue,
        match_radius_arcsec=args.match_radius_arcsec,
        spectral_index=args.spectral_index,
        temperature_bin_width_c=args.temperature_bin_width_c,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    finite_temperature_bins = np.isfinite(
        np.asarray(temperature_bin_table["Temperature_bin_center_C"], dtype=float)
    )
    print(f"LOW3 rows loaded: {len(racs_catalogue)}")
    print(f"NVSS rows loaded: {len(nvss_catalogue)}")
    print(f"Valid crossmatches saved: {len(matched_table)}")
    print(f"Temperature bins with matches: {len(temperature_bin_table)}")
    print(f"Temperature bins with finite mean PAF temperature: {int(np.sum(finite_temperature_bins))}")
    plot_flux_ratio_vs_temperature(matched_table, temperature_bin_table)
    plt.show()
