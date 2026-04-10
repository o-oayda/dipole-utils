import numpy as np
from astropy.table import Table

from scripts import racs_low3_nvss_flux_ratios


def test_build_crossmatched_flux_ratio_products_adds_tile_temperature(monkeypatch):
    racs_catalogue = Table(
        {
            "Name": ["RACS_A", "RACS_B", "RACS_C"],
            "RA": [10.0, 20.0, 20.01],
            "Dec": [0.0, 1.0, 1.01],
            "Total_flux": [100.0, 200.0, 300.0],
            "SBID": [111, 222, 222],
            "Scan_start_MJD": [60000.0, 60001.0, 60001.0],
        }
    )
    nvss_catalogue = Table(
        {
            "source_name": ["NVSS_A", "NVSS_B"],
            "ra": [10.0, 20.0],
            "dec": [0.0, 1.0],
            "integrated_flux": [80.0, 160.0],
        }
    )

    def fake_get_mean_paf_temperatures_for_mjd(mjd_values, data_dir=None, max_time_offset_minutes=5.0):
        mjd_values = np.asarray(mjd_values, dtype=float)
        return mjd_values - 59990.0

    monkeypatch.setattr(
        racs_low3_nvss_flux_ratios,
        "get_mean_paf_temperatures_for_mjd",
        fake_get_mean_paf_temperatures_for_mjd,
    )

    matched_table, temperature_bin_table, ratio_payload = (
        racs_low3_nvss_flux_ratios.build_crossmatched_flux_ratio_products(
            racs_catalogue,
            nvss_catalogue,
            match_radius_arcsec=5.0,
            bootstrap_resamples=32,
        )
    )

    expected_scaled_nvss = racs_low3_nvss_flux_ratios.scale_flux_density([80.0, 160.0])
    expected_ratio = np.asarray([100.0, 200.0]) / expected_scaled_nvss

    assert len(matched_table) == 2
    assert np.allclose(
        np.asarray(temperature_bin_table["Temperature_bin_start_C"], dtype=float),
        [10.0, 11.0],
    )
    assert np.allclose(
        np.asarray(temperature_bin_table["Temperature_bin_center_C"], dtype=float),
        [10.5, 11.5],
    )
    assert np.array_equal(np.asarray(temperature_bin_table["N_Tiles"]), np.asarray([1, 1]))
    assert np.array_equal(np.asarray(temperature_bin_table["N_Matches"]), np.asarray([1, 1]))
    assert np.allclose(
        np.asarray(temperature_bin_table["Std_Flux_ratio_RACS_over_NVSS_scaled"], dtype=float),
        [0.0, 0.0],
    )
    assert np.allclose(
        np.asarray(
            temperature_bin_table["Bootstrap_uncertainty_on_mean_Flux_ratio"],
            dtype=float,
        ),
        [0.0],
    )
    assert np.allclose(
        np.asarray(
            temperature_bin_table["Mean_Flux_ratio_RACS_over_NVSS_scaled"],
            dtype=float,
        ),
        expected_ratio,
    )
    assert np.allclose(
        np.asarray(matched_table["Scaled_NVSS_flux_943p5MHz"], dtype=float),
        expected_scaled_nvss,
    )
    assert np.allclose(
        np.asarray(matched_table["Flux_ratio_RACS_over_NVSS_scaled"], dtype=float),
        expected_ratio,
    )
    assert np.allclose(
        np.asarray(matched_table["Mean_PAF_Temperature_C"], dtype=float),
        [10.0, 11.0],
    )

    ratio_arrays_by_sbid = ratio_payload["ratio_arrays_by_sbid"]
    assert np.allclose(ratio_arrays_by_sbid[111], [expected_ratio[0]])
    assert np.allclose(ratio_arrays_by_sbid[222], [expected_ratio[1]])
    ratio_arrays_by_temperature_bin = ratio_payload["ratio_arrays_by_temperature_bin"]
    assert np.allclose(ratio_arrays_by_temperature_bin[10.0], [expected_ratio[0]])
    assert np.allclose(ratio_arrays_by_temperature_bin[11.0], [expected_ratio[1]])
