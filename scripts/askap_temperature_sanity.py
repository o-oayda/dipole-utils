from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np

from dipoleutils.utils.weather import get_hourly_temperatures_for_date


SUMMER_DATE = "2023-01-15"
WINTER_DATE = "2023-07-15"


def _utc_seconds_to_hour_of_day(unix_seconds: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (
                datetime_value.hour
                + datetime_value.minute / 60
                + datetime_value.second / 3600
            )
            for datetime_value in (
                datetime.fromtimestamp(float(time_value), tz=UTC)
                for time_value in unix_seconds
            )
        ],
        dtype=float,
    )


summer_unix, summer_temp = get_hourly_temperatures_for_date(SUMMER_DATE)
winter_unix, winter_temp = get_hourly_temperatures_for_date(WINTER_DATE)

summer_hours = _utc_seconds_to_hour_of_day(summer_unix)
winter_hours = _utc_seconds_to_hour_of_day(winter_unix)

plt.figure(figsize=(10, 5))
plt.plot(summer_hours, summer_temp, marker="o", markersize=3, label=SUMMER_DATE)
plt.plot(winter_hours, winter_temp, marker="o", markersize=3, label=WINTER_DATE)
plt.xlim(0, 23)
plt.xticks(np.arange(24), [f"{hour:02d}:00" for hour in range(24)], rotation=45)
plt.xlabel("UTC Time")
plt.ylabel("Temperature (C)")
plt.title("ASKAP Site Hourly Temperature Sanity Check")
plt.legend()
plt.tight_layout()
plt.show()
