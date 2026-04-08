from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
import csv
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PAF_TEMPERATURE_DIR = Path(__file__).resolve().parents[1] / "data" / "paf_temps"
SECONDS_PER_DAY = 86400.0
MJD_UNIX_EPOCH_OFFSET_DAYS = 40587.0
ANTENNA_NAME_PATTERN = re.compile(r"(ak\d{2})")
ASKAP_LOCAL_UTC_OFFSET_HOURS = 8.0
DEFAULT_PAF_MEAN_CADENCE_MINUTES = 10.0


@dataclass(frozen=True)
class PafTemperatureMatch:
    antenna_names: tuple[str, ...]
    temperatures_c: np.ndarray
    matched_time_offsets_seconds: np.ndarray
    matched_unix_seconds: np.ndarray


def _mjd_to_unix_seconds(mjd_value: float) -> float:
    return (float(mjd_value) - MJD_UNIX_EPOCH_OFFSET_DAYS) * SECONDS_PER_DAY


def _parse_antenna_name(path: Path) -> str:
    match = ANTENNA_NAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse ASKAP antenna name from {path}.")
    return match.group(1)


def _load_single_temperature_series(
    csv_path: Path,
    utc_offset_hours: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    valid_unix_seconds: list[float] = []
    valid_temperatures: list[float] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) < 2 or row[1] == "":
                continue
            timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            valid_unix_seconds.append(
                timestamp.timestamp() - utc_offset_hours * 3600.0
            )
            valid_temperatures.append(float(row[1]))

    return (
        np.asarray(valid_unix_seconds, dtype=float),
        np.asarray(valid_temperatures, dtype=float),
    )

def _load_single_paf_temperature_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # The antenna exports appear to be in ASKAP local time (UTC+8), unlike the
    # ambient file, which already lines up as UTC.
    return _load_single_temperature_series(
        csv_path,
        utc_offset_hours=ASKAP_LOCAL_UTC_OFFSET_HOURS,
    )


def _load_ambient_temperature_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_single_temperature_series(csv_path, utc_offset_hours=0.0)


def _normalise_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir).expanduser().resolve()


@lru_cache(maxsize=None)
def _load_paf_temperature_series_cached(
    data_dir_str: str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    data_dir = Path(data_dir_str)
    csv_paths = sorted(data_dir.glob("ak*csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per-antenna PAF temperature files found in {data_dir}.")

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for csv_path in csv_paths:
        antenna_name = _parse_antenna_name(csv_path)
        unix_seconds, temperatures_c = _load_single_paf_temperature_series(csv_path)
        series.append((antenna_name, unix_seconds, temperatures_c))

    antenna_names = [item[0] for item in series]
    if len(antenna_names) != 36:
        raise ValueError(
            f"Expected 36 ASKAP antenna temperature files, found {len(antenna_names)} in {data_dir}."
        )
    if antenna_names != [f"ak{antenna_index:02d}" for antenna_index in range(1, 37)]:
        raise ValueError("PAF temperature files do not cover the expected ak01-ak36 antennas.")

    return tuple(series)


@lru_cache(maxsize=None)
def _load_ambient_temperature_series_cached(data_dir_str: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir_str)
    csv_paths = sorted(data_dir.glob("Temperature & Humidity-data-*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No ambient temperature file found in {data_dir}.")
    if len(csv_paths) > 1:
        raise ValueError(f"Expected one ambient temperature file in {data_dir}, found {len(csv_paths)}.")
    return _load_ambient_temperature_series(csv_paths[0])


def load_paf_temperature_series(
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    return _load_paf_temperature_series_cached(str(_normalise_data_dir(data_dir)))


def load_ambient_temperature_series(
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    return _load_ambient_temperature_series_cached(str(_normalise_data_dir(data_dir)))


def _unix_seconds_to_hour_of_day(unix_seconds: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
            for timestamp in (
                datetime.fromtimestamp(float(time_value), tz=UTC)
                for time_value in unix_seconds
            )
        ],
        dtype=float,
    )


def _unix_seconds_to_local_hour_of_day(unix_seconds: np.ndarray) -> np.ndarray:
    return np.mod(
        _unix_seconds_to_hour_of_day(unix_seconds) + ASKAP_LOCAL_UTC_OFFSET_HOURS,
        24.0,
    )


def get_paf_temperatures_for_mjd(
    obs_mjd: float,
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
    max_time_offset_minutes: float = 5.0,
) -> PafTemperatureMatch:
    """
    Return one temperature per ASKAP antenna for a single observation timestamp.

    The PAF exports are sparse: valid readings appear every several minutes, with
    blank rows filling the 1-minute grid. This function therefore matches each
    antenna to the nearest valid sample within ``max_time_offset_minutes`` and
    returns ``NaN`` where no sample is close enough.
    """
    target_unix_seconds = _mjd_to_unix_seconds(obs_mjd)
    max_time_offset_seconds = float(max_time_offset_minutes) * 60.0

    antenna_names: list[str] = []
    temperatures_c: list[float] = []
    matched_time_offsets_seconds: list[float] = []
    matched_unix_seconds: list[float] = []

    for antenna_name, unix_seconds, antenna_temperatures in load_paf_temperature_series(data_dir):
        antenna_names.append(antenna_name)
        if unix_seconds.size == 0:
            temperatures_c.append(np.nan)
            matched_time_offsets_seconds.append(np.nan)
            matched_unix_seconds.append(np.nan)
            continue

        insertion_index = int(np.searchsorted(unix_seconds, target_unix_seconds))
        candidate_indices: list[int] = []
        if insertion_index < unix_seconds.size:
            candidate_indices.append(insertion_index)
        if insertion_index > 0:
            candidate_indices.append(insertion_index - 1)

        best_index = min(
            candidate_indices,
            key=lambda index: abs(unix_seconds[index] - target_unix_seconds),
        )
        best_offset_seconds = abs(unix_seconds[best_index] - target_unix_seconds)
        if best_offset_seconds > max_time_offset_seconds:
            temperatures_c.append(np.nan)
            matched_time_offsets_seconds.append(np.nan)
            matched_unix_seconds.append(np.nan)
            continue

        temperatures_c.append(float(antenna_temperatures[best_index]))
        matched_time_offsets_seconds.append(float(best_offset_seconds))
        matched_unix_seconds.append(float(unix_seconds[best_index]))

    return PafTemperatureMatch(
        antenna_names=tuple(antenna_names),
        temperatures_c=np.asarray(temperatures_c, dtype=float),
        matched_time_offsets_seconds=np.asarray(matched_time_offsets_seconds, dtype=float),
        matched_unix_seconds=np.asarray(matched_unix_seconds, dtype=float),
    )


def get_mean_paf_temperature_for_mjd(
    obs_mjd: float,
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
    max_time_offset_minutes: float = 5.0,
) -> float:
    match = get_paf_temperatures_for_mjd(
        obs_mjd,
        data_dir=data_dir,
        max_time_offset_minutes=max_time_offset_minutes,
    )
    finite_temperatures = match.temperatures_c[np.isfinite(match.temperatures_c)]
    if finite_temperatures.size == 0:
        return float("nan")
    return float(np.mean(finite_temperatures, dtype=float))


def get_mean_paf_temperatures_for_mjd(
    mjd_values: Iterable[float],
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
    max_time_offset_minutes: float = 5.0,
) -> np.ndarray:
    mjd_array = np.asarray(mjd_values, dtype=float)
    if mjd_array.size == 0:
        return np.asarray([], dtype=float)

    unique_mjd, inverse_indices = np.unique(mjd_array, return_inverse=True)
    mean_temperatures = np.asarray(
        [
            get_mean_paf_temperature_for_mjd(
                obs_mjd,
                data_dir=data_dir,
                max_time_offset_minutes=max_time_offset_minutes,
            )
            for obs_mjd in unique_mjd
        ],
        dtype=float,
    )
    return mean_temperatures[inverse_indices]


def plot_paf_temperatures_for_day(
    date_str: str,
    data_dir: Path | str = DEFAULT_PAF_TEMPERATURE_DIR,
) -> None:
    """
    Plot all antenna PAF temperatures for one ASKAP local day.

    Parameters
    ----------
    date_str
        ASKAP local calendar date in ``YYYY-MM-DD`` format.
    """
    day_start = (
        datetime.fromisoformat(date_str).replace(tzinfo=UTC).timestamp()
        - ASKAP_LOCAL_UTC_OFFSET_HOURS * 3600.0
    )
    day_end = day_start + SECONDS_PER_DAY

    figure, axis = plt.subplots(figsize=(11, 5))
    day_grid_unix_seconds = np.arange(
        day_start,
        day_end,
        DEFAULT_PAF_MEAN_CADENCE_MINUTES * 60.0,
        dtype=float,
    )
    gridded_antenna_temperatures: list[np.ndarray] = []
    for antenna_name, unix_seconds, temperatures_c in load_paf_temperature_series(data_dir):
        in_day = (unix_seconds >= day_start) & (unix_seconds < day_end)
        if not np.any(in_day):
            continue
        day_unix_seconds = unix_seconds[in_day]
        day_temperatures_c = temperatures_c[in_day]
        axis.plot(
            _unix_seconds_to_local_hour_of_day(day_unix_seconds),
            day_temperatures_c,
            linewidth=1.0,
            alpha=0.8,
            label=antenna_name,
        )
        antenna_grid_temperatures = np.full(day_grid_unix_seconds.shape, np.nan, dtype=float)
        for grid_index, grid_unix_seconds in enumerate(day_grid_unix_seconds):
            insertion_index = int(np.searchsorted(day_unix_seconds, grid_unix_seconds))
            candidate_indices: list[int] = []
            if insertion_index < day_unix_seconds.size:
                candidate_indices.append(insertion_index)
            if insertion_index > 0:
                candidate_indices.append(insertion_index - 1)
            if not candidate_indices:
                continue
            best_index = min(
                candidate_indices,
                key=lambda index: abs(day_unix_seconds[index] - grid_unix_seconds),
            )
            best_offset_seconds = abs(day_unix_seconds[best_index] - grid_unix_seconds)
            if best_offset_seconds <= DEFAULT_PAF_MEAN_CADENCE_MINUTES * 30.0:
                antenna_grid_temperatures[grid_index] = float(day_temperatures_c[best_index])
        gridded_antenna_temperatures.append(antenna_grid_temperatures)

    if gridded_antenna_temperatures:
        mean_temperatures_c = np.nanmean(
            np.vstack(gridded_antenna_temperatures),
            axis=0,
        )
        axis.plot(
            _unix_seconds_to_local_hour_of_day(day_grid_unix_seconds),
            mean_temperatures_c,
            color="tab:red",
            linewidth=2.5,
            label="mean PAF",
            zorder=4,
        )

    ambient_unix_seconds, ambient_temperatures_c = load_ambient_temperature_series(data_dir)
    ambient_in_day = (ambient_unix_seconds >= day_start) & (ambient_unix_seconds < day_end)
    if np.any(ambient_in_day):
        axis.plot(
            _unix_seconds_to_local_hour_of_day(ambient_unix_seconds[ambient_in_day]),
            ambient_temperatures_c[ambient_in_day],
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="ambient",
            zorder=5,
        )

    axis.set_xlim(0, 24)
    axis.set_xticks(np.arange(0, 25, 2))
    axis.set_xlabel("Time (UTC+8)")
    axis.set_ylabel("Temperature (C)")
    axis.set_title(f"ASKAP PAF Temperatures on {date_str}")
    axis.legend(ncol=4, fontsize=8)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_paf_temperatures_for_day("2023-12-27")
