from __future__ import annotations

import copy
import inspect
import warnings
from importlib import resources
from typing import Any, Mapping

import numpy as np
import yaml
from astropy.table import Table
from astropy.time import Time
from numpy.typing import NDArray

from .data_loader import DataLoader
from .mask import Masker
from .samples import CatalogueToMap
from .weather import get_temperatures_for_mjd


RACS_DEFAULTS_RESOURCE = "racs_defaults.yaml"
ASKAP_UTC_OFFSET_HOURS = 8.0
MASK_METHODS = (
    "mask_galactic_plane",
    "mask_equatorial_poles",
    "mask_ecliptic_poles",
    "mask_a_team_sources",
    "mask_slice",
    "mask_equatorial_longitude",
    "mask_around_bright_sources",
)
CATALOGUE_OPERATIONS = (
    "crossmatch_local_sources",
)
VALID_COORDINATE_SYSTEMS = {"equatorial", "galactic", "ecliptic"}
REQUIRED_DEFAULT_KEYS = {
    "flux_column",
    "flux_min",
    "flux_max",
    "nside",
    "coordinate_system",
    "nest",
    "masks",
    "catalogue_operations",
}


def load_racs_defaults() -> dict[str, dict[str, Any]]:
    """Load packaged fiducial defaults for RACS catalogue variants."""
    defaults_path = resources.files("dipoleutils.data").joinpath(
        RACS_DEFAULTS_RESOURCE
    )
    with defaults_path.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle)
    if not isinstance(defaults, dict):
        raise TypeError("Packaged RACS defaults must be a mapping.")
    return defaults


def _deep_merge(
        base: dict[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key == "masks"
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = copy.deepcopy(merged[key])
            for method_name, call_spec in value.items():
                merged[key][method_name] = copy.deepcopy(call_spec)
            continue
        if (
            key == "catalogue_operations"
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = copy.deepcopy(merged[key])
            for method_name, call_spec in value.items():
                merged[key][method_name] = copy.deepcopy(call_spec)
            continue
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class RACS:
    """
    Convenience wrapper for loading RACS catalogues and making fiducial maps.

    The packaged defaults define flux cuts, map settings, and mask method calls
    per RACS variant. Instance-level overrides are supported with
    :meth:`override_defaults` and never mutate the packaged defaults.
    """

    def __init__(self, variant: str) -> None:
        self.variant = variant
        all_defaults = load_racs_defaults()
        if variant not in all_defaults:
            available = ", ".join(sorted(all_defaults))
            raise ValueError(
                f"RACS variant '{variant}' has no fiducial defaults. "
                f"Available variants: {available}"
            )

        self.defaults = copy.deepcopy(all_defaults[variant])
        self.catalogue = DataLoader("racs", variant).load()
        if not isinstance(self.catalogue, Table):
            raise TypeError("RACS catalogues must load as astropy Table objects.")
        self._validate_defaults(self.defaults)

        self.mapper: CatalogueToMap | None = None
        self.masker: Masker | None = None
        self.mask_map: NDArray[np.int64] | None = None
        self.parameter_maps: dict[str, NDArray[np.float64]] = {}

    def override_defaults(self, **overrides: Any) -> None:
        """
        Apply instance-local fiducial default overrides.

        Examples:
            ``racs.override_defaults(flux_min=20, flux_max=None)``
            ``racs.override_defaults(masks={"mask_equatorial_poles": {...}})``
        """
        updated = _deep_merge(self.defaults, overrides)
        self._validate_defaults(updated)
        self.defaults = updated

    def set_flux_cut(
            self,
            minimum: float | None = None,
            maximum: float | None = None,
        ) -> None:
        """Set the fiducial flux cut for this instance."""
        self.override_defaults(flux_min=minimum, flux_max=maximum)

    def set_equatorial_pole_mask(
            self,
            north_radius: float = 0.0,
            south_radius: float = 0.0,
        ) -> None:
        """Set the equatorial pole mask radii for this instance."""
        self.set_mask(
            "mask_equatorial_poles",
            {"north_radius": north_radius, "south_radius": south_radius},
        )

    def set_mask(self, method_name: str, call_spec: Any) -> None:
        """Set a single mask method config for this instance."""
        self.override_defaults(masks={method_name: call_spec})

    def disable_mask(self, method_name: str) -> None:
        """Disable a configured mask method for this instance."""
        self.set_mask(method_name, None)

    def remove_local_sources(
            self,
            radius: float,
            coordinate_system: str | None = None,
            source_name_A_column: str | None = None,
        ) -> None:
        """
        Remove catalogue sources cross-matched to the packaged local-source list.

        The operation is applied after the flux cut and before density map
        creation.
        """
        if coordinate_system is None:
            coordinate_system = self.defaults["coordinate_system"]
        self.set_catalogue_operation(
            "crossmatch_local_sources",
            {
                "coordinate_system": coordinate_system,
                "radius": radius,
                "source_name_A_column": source_name_A_column,
            },
        )

    def set_catalogue_operation(self, method_name: str, call_spec: Any) -> None:
        """Set a catalogue-level operation to run before density map creation."""
        self.override_defaults(catalogue_operations={method_name: call_spec})

    def disable_catalogue_operation(self, method_name: str) -> None:
        """Disable a configured catalogue-level operation."""
        self.set_catalogue_operation(method_name, None)

    def get_mask_config(
            self,
            active_only: bool = False,
        ) -> dict[str, Any]:
        """Return this instance's mask configuration."""
        masks = copy.deepcopy(self.defaults["masks"])
        if not active_only:
            return masks
        return {
            method_name: call_spec
            for method_name, call_spec in masks.items()
            if call_spec is not None
        }

    def get_catalogue_operations(
            self,
            active_only: bool = False,
        ) -> dict[str, Any]:
        """Return configured catalogue-level operations."""
        operations = copy.deepcopy(self.defaults["catalogue_operations"])
        if not active_only:
            return operations
        return {
            method_name: call_spec
            for method_name, call_spec in operations.items()
            if call_spec is not None
        }

    def describe_fiducial_map(
            self,
            active_masks_only: bool = True,
        ) -> str:
        """Return a readable description of this instance's fiducial map setup."""
        flux_min = self.defaults.get("flux_min")
        flux_max = self.defaults.get("flux_max")
        flux_cut = f"{flux_min} <= {self.defaults['flux_column']} <= {flux_max}"
        if flux_min is None and flux_max is None:
            flux_cut = "none"
        elif flux_min is None:
            flux_cut = f"{self.defaults['flux_column']} <= {flux_max}"
        elif flux_max is None:
            flux_cut = f"{self.defaults['flux_column']} >= {flux_min}"

        lines = [
            f"RACS variant: {self.variant}",
            f"Flux cut: {flux_cut}",
            (
                "Map: "
                f"nside={self.defaults['nside']}, "
                f"coordinate_system={self.defaults['coordinate_system']}, "
                f"nest={self.defaults['nest']}"
            ),
            "Catalogue operations:",
        ]
        operations = self.get_catalogue_operations(active_only=True)
        if not operations:
            lines.append("  none")
        else:
            for method_name, call_spec in operations.items():
                lines.append(f"  {method_name}: {call_spec}")

        lines.extend([
            "Masks:",
        ])
        masks = self.get_mask_config(active_only=active_masks_only)
        if not masks:
            lines.append("  none")
        else:
            for method_name, call_spec in masks.items():
                lines.append(f"  {method_name}: {call_spec}")
        return "\n".join(lines)

    def show_defaults(self, active_masks_only: bool = True) -> None:
        """Print a readable description of this instance's fiducial map setup."""
        print(self.describe_fiducial_map(active_masks_only=active_masks_only))

    def add_temperature_columns(
            self,
            time_column: str | None = None,
            temperature_column: str = "Temperature_C",
            start_time_column: str = "Start_time_hours",
            date_column: str = "Scan_start_date",
            askap_utc_offset_hours: float = ASKAP_UTC_OFFSET_HOURS,
            prefer_paf: bool = True,
        ) -> None:
        """
        Add source temperature and ASKAP local start-time columns.

        Mean PAF temperatures are tried first. If PAF lookup raises or returns
        no finite temperatures, Open-Meteo temperatures are used as a fallback.
        """
        if time_column is None:
            time_column = self._infer_time_column()
        if time_column not in self.catalogue.colnames:
            raise ValueError(
                f"RACS time column '{time_column}' is not in the loaded "
                f"{self.variant} catalogue."
            )

        mjd_values = np.asarray(self.catalogue[time_column], dtype=float)
        self.catalogue[start_time_column] = np.mod(
            mjd_values % 1.0 * 24.0 + askap_utc_offset_hours,
            24.0,
        )
        self.catalogue[date_column] = Time(mjd_values, format="mjd").iso
        self.catalogue[temperature_column] = self._lookup_temperatures(
            mjd_values,
            prefer_paf=prefer_paf,
        )

    def make_parameter_map(
            self,
            column_name: str | list[str],
            operation: str = "mean",
            coordinate_system: str | None = None,
            nside: int | None = None,
            nest: bool | None = None,
            no_source_val: float = np.nan,
            apply_fiducial_mask: bool = True,
        ) -> NDArray[np.float64] | list:
        """
        Create a parameter map using the same fiducial catalogue cuts.

        When ``apply_fiducial_mask`` is true, configured RACS masks are applied
        to the returned map.
        """
        mapper = self._make_prepared_mapper()
        coordinate_system = coordinate_system or self.defaults["coordinate_system"]
        nside = self.defaults["nside"] if nside is None else nside
        nest = self.defaults["nest"] if nest is None else nest

        parameter_map = mapper.make_parameter_map(
            column_name=column_name,
            coordinate_system=coordinate_system,
            no_source_val=no_source_val,
            nside=nside,
            nest=nest,
            operation=operation,  # type: ignore[arg-type]
        )
        if apply_fiducial_mask:
            masker = Masker(parameter_map, coordinate_system=coordinate_system)
            self._apply_masks(masker)
            parameter_map = masker.get_masked_density_map()

        map_key = (
            column_name
            if isinstance(column_name, str)
            else f"{operation}({', '.join(column_name)})"
        )
        if isinstance(parameter_map, np.ndarray):
            self.parameter_maps[map_key] = parameter_map
        self.mapper = mapper
        return parameter_map

    def make_fiducial_map(self) -> NDArray[np.float64]:
        """Create and return the fiducial NaN-masked RACS density map."""
        mapper = self._make_prepared_mapper()
        coordinate_system = self.defaults.get("coordinate_system", "equatorial")
        nside = self.defaults.get("nside", 64)
        nest = self.defaults.get("nest", False)
        density_map = mapper.make_density_map(
            coordinate_system=coordinate_system,
            nside=nside,
            nest=nest,
        )

        masker = Masker(density_map, coordinate_system=coordinate_system)
        self._apply_masks(masker)

        self.mapper = mapper
        self.masker = masker
        self.mask_map = masker.get_mask_map()
        return masker.get_masked_density_map()

    def _make_prepared_mapper(self) -> CatalogueToMap:
        self._validate_defaults(self.defaults)
        mapper = CatalogueToMap(self.catalogue.copy(copy_data=True))

        flux_column = self.defaults["flux_column"]
        flux_min = self.defaults.get("flux_min")
        flux_max = self.defaults.get("flux_max")
        if flux_min is not None or flux_max is not None:
            mapper.make_cut(flux_column, minimum=flux_min, maximum=flux_max)

        self._apply_catalogue_operations(mapper)
        return mapper

    def _infer_time_column(self) -> str:
        preferred_columns = (
            "Scan_start_MJD",
            "obs_start_time",
            "scan_start_mjd",
        )
        for column_name in preferred_columns:
            if column_name in self.catalogue.colnames:
                return column_name
        raise ValueError(
            "Could not infer a RACS observation-time column. Pass "
            "time_column explicitly."
        )

    def _lookup_temperatures(
            self,
            mjd_values: NDArray[np.float64],
            prefer_paf: bool = True,
        ) -> NDArray[np.float64]:
        if prefer_paf:
            try:
                from scripts.paf_temperature_lookup import (  # type: ignore[import-not-found]
                    get_mean_paf_temperatures_for_mjd,
                )

                temperatures = np.asarray(
                    get_mean_paf_temperatures_for_mjd(mjd_values),
                    dtype=float,
                )
                if np.any(np.isfinite(temperatures)):
                    return temperatures
                raise RuntimeError("PAF lookup returned no finite temperatures.")
            except Exception as exc:
                warnings.warn(
                    "Unable to use PAF temperature data; falling back to "
                    f"Open-Meteo: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return np.asarray(get_temperatures_for_mjd(mjd_values), dtype=float)

    def _apply_masks(self, masker: Masker) -> None:
        masks = self.defaults.get("masks", {})
        if not isinstance(masks, Mapping):
            raise TypeError("RACS masks configuration must be a mapping.")

        for method_name, call_spec in masks.items():
            if call_spec is None:
                continue
            if method_name not in MASK_METHODS:
                raise ValueError(f"Unknown RACS mask method: {method_name}")
            method = getattr(masker, method_name)
            for args, kwargs in self._iter_mask_calls(method_name, call_spec):
                method(*args, **kwargs)

    def _apply_catalogue_operations(self, mapper: CatalogueToMap) -> None:
        operations = self.defaults.get("catalogue_operations", {})
        if not isinstance(operations, Mapping):
            raise TypeError("RACS catalogue_operations must be a mapping.")

        for method_name, call_spec in operations.items():
            if call_spec is None:
                continue
            if method_name not in CATALOGUE_OPERATIONS:
                raise ValueError(f"Unknown RACS catalogue operation: {method_name}")
            method = getattr(mapper, method_name)
            for args, kwargs in self._iter_call_specs(method_name, call_spec):
                method(*args, **kwargs)

    def _validate_defaults(self, defaults: Mapping[str, Any]) -> None:
        missing = REQUIRED_DEFAULT_KEYS - set(defaults)
        if missing:
            raise ValueError(
                f"RACS defaults missing required key(s): {sorted(missing)}"
            )

        flux_column = defaults["flux_column"]
        if not isinstance(flux_column, str):
            raise TypeError("RACS flux_column must be a string.")
        if flux_column not in self.catalogue.colnames:
            raise ValueError(
                f"RACS flux column '{flux_column}' is not in the loaded "
                f"{self.variant} catalogue."
            )

        for bound_name in ("flux_min", "flux_max"):
            bound = defaults.get(bound_name)
            if bound is not None and not isinstance(bound, (int, float)):
                raise TypeError(f"RACS {bound_name} must be numeric or None.")

        flux_min = defaults.get("flux_min")
        flux_max = defaults.get("flux_max")
        if (
            flux_min is not None
            and flux_max is not None
            and float(flux_min) > float(flux_max)
        ):
            raise ValueError("RACS flux_min must be <= flux_max.")

        nside = defaults["nside"]
        if (
            not isinstance(nside, int)
            or nside <= 0
            or nside & (nside - 1) != 0
        ):
            raise ValueError("RACS nside must be a positive power of two.")

        coordinate_system = defaults["coordinate_system"]
        if coordinate_system not in VALID_COORDINATE_SYSTEMS:
            allowed = ", ".join(sorted(VALID_COORDINATE_SYSTEMS))
            raise ValueError(
                f"RACS coordinate_system must be one of: {allowed}."
            )
        if not isinstance(defaults["nest"], bool):
            raise TypeError("RACS nest must be a bool.")

        masks = defaults["masks"]
        if not isinstance(masks, Mapping):
            raise TypeError("RACS masks configuration must be a mapping.")
        for method_name, call_spec in masks.items():
            if method_name not in MASK_METHODS:
                raise ValueError(f"Unknown RACS mask method: {method_name}")
            if call_spec is None:
                continue
            for args, kwargs in self._iter_mask_calls(method_name, call_spec):
                self._validate_mask_call(method_name, args, kwargs)

        operations = defaults["catalogue_operations"]
        if not isinstance(operations, Mapping):
            raise TypeError("RACS catalogue_operations must be a mapping.")
        for method_name, call_spec in operations.items():
            if method_name not in CATALOGUE_OPERATIONS:
                raise ValueError(f"Unknown RACS catalogue operation: {method_name}")
            if call_spec is None:
                continue
            for args, kwargs in self._iter_call_specs(method_name, call_spec):
                self._validate_catalogue_operation_call(method_name, args, kwargs)

    @staticmethod
    def _validate_mask_call(
            method_name: str,
            args: list[Any],
            kwargs: dict[str, Any],
        ) -> None:
        method = getattr(Masker, method_name)
        signature = inspect.signature(method)
        try:
            signature.bind(object(), *args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid RACS config for {method_name}: {exc}"
            ) from exc

    @staticmethod
    def _validate_catalogue_operation_call(
            method_name: str,
            args: list[Any],
            kwargs: dict[str, Any],
        ) -> None:
        method = getattr(CatalogueToMap, method_name)
        signature = inspect.signature(method)
        try:
            signature.bind(object(), *args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid RACS config for catalogue operation "
                f"{method_name}: {exc}"
            ) from exc

    @classmethod
    def _iter_mask_calls(
            cls,
            method_name: str,
            call_spec: Any,
        ) -> list[tuple[list[Any], dict[str, Any]]]:
        return cls._iter_call_specs(method_name, call_spec)

    @staticmethod
    def _iter_call_specs(
            method_name: str,
            call_spec: Any,
        ) -> list[tuple[list[Any], dict[str, Any]]]:
        specs = call_spec if isinstance(call_spec, list) else [call_spec]
        calls = []
        for spec in specs:
            if isinstance(spec, tuple):
                spec = list(spec)
            if isinstance(spec, list):
                calls.append((spec, {}))
                continue
            if not isinstance(spec, Mapping):
                raise TypeError(
                    f"RACS mask config for {method_name} must be a mapping "
                    "or list of positional arguments."
                )
            if "args" in spec or "kwargs" in spec:
                args = spec.get("args", [])
                kwargs = spec.get("kwargs", {})
            else:
                args = []
                kwargs = spec
            if not isinstance(args, list):
                raise TypeError(f"RACS mask args for {method_name} must be a list.")
            if not isinstance(kwargs, Mapping):
                raise TypeError(
                    f"RACS mask kwargs for {method_name} must be a mapping."
                )
            calls.append((args, dict(kwargs)))
        return calls
