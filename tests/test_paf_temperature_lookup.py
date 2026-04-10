from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.time import Time

from scripts.paf_temperature_lookup import (
    get_mean_paf_temperature_for_mjd,
    get_mean_paf_temperatures_for_mjd,
    get_paf_temperatures_for_mjd,
)


def _write_paf_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "\n".join(
            [
                '"Time","temperature"',
                *[f"{timestamp},{value}" for timestamp, value in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_full_antenna_set(tmp_path: Path, minute_offset: int = 0) -> None:
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    for antenna_index in range(1, 37):
        antenna_name = f"ak{antenna_index:02d}"
        _write_paf_csv(
            tmp_path / f"{antenna_name} ctrl_adc1_pafAvTemp-data.csv",
            [
                (base_time.strftime("%Y-%m-%d %H:%M:%S"), f"{antenna_index:.1f}"),
                (
                    (base_time + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                    f"{antenna_index + 100:.1f}",
                ),
            ],
        )


def test_get_paf_temperatures_for_mjd_returns_all_36_temperatures(tmp_path):
    _make_full_antenna_set(tmp_path)
    obs_mjd = Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd

    result = get_paf_temperatures_for_mjd(
        obs_mjd,
        data_dir=tmp_path,
        max_interpolation_gap_minutes=20.0,
    )

    assert result.antenna_names == tuple(f"ak{antenna_index:02d}" for antenna_index in range(1, 37))
    assert result.temperatures_c.shape == (36,)
    assert np.allclose(result.temperatures_c, np.arange(41.0, 77.0))
    assert np.allclose(result.matched_time_offsets_seconds, np.full(36, 240.0))


def test_get_paf_temperatures_for_mjd_marks_large_interpolation_gaps_as_nan(tmp_path):
    _make_full_antenna_set(tmp_path, minute_offset=30)
    obs_mjd = Time(datetime(2023, 12, 31, 16, 0, tzinfo=UTC)).mjd

    result = get_paf_temperatures_for_mjd(
        obs_mjd,
        data_dir=tmp_path,
        max_interpolation_gap_minutes=20.0,
    )

    assert np.all(np.isnan(result.temperatures_c))
    assert np.all(np.isnan(result.matched_time_offsets_seconds))
    assert np.all(np.isnan(result.matched_unix_seconds))


def test_get_mean_paf_temperature_for_mjd_averages_finite_antenna_matches(tmp_path):
    _make_full_antenna_set(tmp_path)
    obs_mjd = Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd

    mean_temperature = get_mean_paf_temperature_for_mjd(
        obs_mjd,
        data_dir=tmp_path,
        max_interpolation_gap_minutes=20.0,
    )

    assert np.isclose(mean_temperature, np.mean(np.arange(41.0, 77.0)))


def test_get_mean_paf_temperatures_for_mjd_reuses_unique_timestamps(tmp_path):
    _make_full_antenna_set(tmp_path)
    repeated_mjd = Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd

    temperatures = get_mean_paf_temperatures_for_mjd(
        [repeated_mjd, repeated_mjd],
        data_dir=tmp_path,
        max_interpolation_gap_minutes=20.0,
    )

    assert np.allclose(
        temperatures,
        np.full(2, np.mean(np.arange(41.0, 77.0))),
    )
