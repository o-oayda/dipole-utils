from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.time import Time

from dipoleutils.utils import weather
from dipoleutils.utils.weather import get_temperatures_for_mjd


class _MockResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args, **kwargs):
        return self.payload


@patch("dipoleutils.utils.weather.urlopen")
def test_get_temperatures_for_mjd_interpolates_hourly_temperatures(mock_urlopen, tmp_path):
    mock_urlopen.return_value = _MockResponse(
        b"""
        {
            "hourly": {
                "time": [
                    "2024-01-01T00:00",
                    "2024-01-01T01:00",
                    "2024-01-01T02:00"
                ],
                "temperature_2m": [10.0, 14.0, 18.0]
            }
        }
        """
    )

    mjd_values = Time(
        [
            datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 1, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 2, 0, tzinfo=UTC),
        ]
    ).mjd

    with patch.object(weather, "WEATHER_CACHE_DIR", Path(tmp_path)):
        temperatures = get_temperatures_for_mjd(mjd_values)

    assert np.allclose(temperatures, np.asarray([14.0, 16.0, 18.0]))
    assert mock_urlopen.call_count == 1


@patch("dipoleutils.utils.weather.urlopen")
def test_get_temperatures_for_mjd_returns_empty_array_for_empty_input(mock_urlopen):
    temperatures = get_temperatures_for_mjd([])

    assert temperatures.size == 0
    mock_urlopen.assert_not_called()


@patch("dipoleutils.utils.weather.urlopen")
def test_get_temperatures_for_mjd_reuses_cached_daily_weather(mock_urlopen, tmp_path):
    mock_urlopen.return_value = _MockResponse(
        b"""
        {
            "hourly": {
                "time": [
                    "2024-01-01T00:00",
                    "2024-01-01T01:00",
                    "2024-01-01T02:00"
                ],
                "temperature_2m": [10.0, 14.0, 18.0]
            }
        }
        """
    )

    mjd_values = Time(
        [
            datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 1, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 1, 30, tzinfo=UTC),
        ]
    ).mjd

    with patch.object(weather, "WEATHER_CACHE_DIR", Path(tmp_path)):
        first_temperatures = get_temperatures_for_mjd(mjd_values)
        second_temperatures = get_temperatures_for_mjd(mjd_values)

    assert np.allclose(first_temperatures, np.asarray([14.0, 16.0, 16.0]))
    assert np.allclose(second_temperatures, np.asarray([14.0, 16.0, 16.0]))
    assert mock_urlopen.call_count == 1
