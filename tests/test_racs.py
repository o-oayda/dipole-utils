from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import Table

from dipoleutils.utils.racs import RACS, load_racs_defaults


def _test_catalogue() -> Table:
    return Table(
        {
            "RA": [0.0, 1.0, 2.0],
            "Dec": [0.0, 1.0, 2.0],
            "Total_flux": [10.0, 20.0, 2000.0],
            "total_flux": [10.0, 20.0, 2000.0],
            "total_flux_source": [10.0, 20.0, 2000.0],
            "Scan_start_MJD": [60000.0, 60000.5, 60001.0],
            "Noise": [1.0, 2.0, 3.0],
        }
    )


def test_racs_defaults_flux_columns() -> None:
    defaults = load_racs_defaults()

    assert defaults["low1"]["flux_column"] == "total_flux_source"
    assert defaults["mid1"]["flux_column"] == "total_flux"

    for variant, config in defaults.items():
        if variant in {"low1", "mid1"}:
            continue
        assert config["flux_column"] == "Total_flux"


def test_make_fiducial_map_calls_core_components() -> None:
    catalogue = _test_catalogue()
    density_map = np.array([1, 2, 3], dtype=np.int_)
    masked_map = np.array([1.0, np.nan, 3.0])

    with (
        patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls,
        patch("dipoleutils.utils.racs.CatalogueToMap") as mapper_cls,
        patch("dipoleutils.utils.racs.Masker") as masker_cls,
    ):
        data_loader_cls.return_value.load.return_value = catalogue
        mapper = MagicMock()
        mapper.make_density_map.return_value = density_map
        mapper_cls.return_value = mapper
        masker = MagicMock()
        masker.get_mask_map.return_value = np.array([1, 0, 1], dtype=np.int64)
        masker.get_masked_density_map.return_value = masked_map
        masker_cls.return_value = masker

        racs = RACS("low3")
        result = racs.make_fiducial_map()

        data_loader_cls.assert_called_once_with("racs", "low3")
    mapper.make_cut.assert_called_once_with(
        "Total_flux",
        minimum=15,
        maximum=1000,
    )
    mapper.make_density_map.assert_called_once_with(
        coordinate_system="equatorial",
        nside=64,
        nest=False,
    )
    masker_cls.assert_called_once_with(density_map, coordinate_system="equatorial")
    masker.mask_galactic_plane.assert_called_once_with(latitude_cut=5)
    masker.mask_equatorial_poles.assert_called_once_with(north_radius=43)
    assert result is masked_map
    assert racs.mask_map is not None


def test_remove_local_sources_runs_before_density_map() -> None:
    catalogue = _test_catalogue()
    density_map = np.array([1, 2, 3], dtype=np.int_)
    call_order = []

    with (
        patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls,
        patch("dipoleutils.utils.racs.CatalogueToMap") as mapper_cls,
        patch("dipoleutils.utils.racs.Masker") as masker_cls,
    ):
        data_loader_cls.return_value.load.return_value = catalogue
        mapper = MagicMock()
        mapper.make_density_map.return_value = density_map
        mapper.crossmatch_local_sources.side_effect = (
            lambda *args, **kwargs: call_order.append("crossmatch")
        )
        mapper.make_density_map.side_effect = (
            lambda *args, **kwargs: call_order.append("density") or density_map
        )
        mapper_cls.return_value = mapper
        masker = MagicMock()
        masker.get_mask_map.return_value = np.array([1, 1, 1], dtype=np.int64)
        masker.get_masked_density_map.return_value = density_map.astype(float)
        masker_cls.return_value = masker

        racs = RACS("low3")
        racs.remove_local_sources(radius=5)
        racs.make_fiducial_map()

    assert racs.defaults["catalogue_operations"]["crossmatch_local_sources"] == {
        "coordinate_system": "equatorial",
        "radius": 5,
        "source_name_A_column": None,
    }
    mapper.crossmatch_local_sources.assert_called_once_with(
        coordinate_system="equatorial",
        radius=5,
        source_name_A_column=None,
    )
    assert call_order == ["crossmatch", "density"]


def test_add_temperature_columns_uses_paf_when_available() -> None:
    with (
        patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls,
        patch(
            "scripts.paf_temperature_lookup.get_mean_paf_temperatures_for_mjd"
        ) as paf_lookup,
        patch("dipoleutils.utils.racs.get_temperatures_for_mjd") as meteo_lookup,
    ):
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        paf_lookup.return_value = np.array([30.0, 31.0, 32.0])
        racs = RACS("low3")
        racs.add_temperature_columns()

    paf_lookup.assert_called_once()
    meteo_lookup.assert_not_called()
    assert np.allclose(racs.catalogue["Temperature_C"], [30.0, 31.0, 32.0])
    assert np.allclose(racs.catalogue["Start_time_hours"], [8.0, 20.0, 8.0])
    assert list(racs.catalogue["Scan_start_date"]) == [
        "2023-02-25 00:00:00.000",
        "2023-02-25 12:00:00.000",
        "2023-02-26 00:00:00.000",
    ]
    assert "Scan_start_JD" not in racs.catalogue.colnames


def test_add_temperature_columns_falls_back_to_open_meteo() -> None:
    with (
        patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls,
        patch(
            "scripts.paf_temperature_lookup.get_mean_paf_temperatures_for_mjd"
        ) as paf_lookup,
        patch("dipoleutils.utils.racs.get_temperatures_for_mjd") as meteo_lookup,
    ):
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        paf_lookup.side_effect = RuntimeError("no PAF data")
        meteo_lookup.return_value = np.array([20.0, 21.0, 22.0])
        racs = RACS("low3")
        with pytest.warns(RuntimeWarning, match="falling back to Open-Meteo"):
            racs.add_temperature_columns()

    meteo_lookup.assert_called_once()
    assert np.allclose(racs.catalogue["Temperature_C"], [20.0, 21.0, 22.0])


def test_make_parameter_map_uses_fiducial_cuts_and_masks() -> None:
    catalogue = _test_catalogue()
    parameter_map = np.array([1.0, 2.0, 3.0])
    masked_map = np.array([1.0, np.nan, 3.0])

    with (
        patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls,
        patch("dipoleutils.utils.racs.CatalogueToMap") as mapper_cls,
        patch("dipoleutils.utils.racs.Masker") as masker_cls,
    ):
        data_loader_cls.return_value.load.return_value = catalogue
        mapper = MagicMock()
        mapper.make_parameter_map.return_value = parameter_map
        mapper_cls.return_value = mapper
        masker = MagicMock()
        masker.get_masked_density_map.return_value = masked_map
        masker_cls.return_value = masker

        racs = RACS("low3")
        result = racs.make_parameter_map("Noise")

    mapper.make_cut.assert_called_once_with(
        "Total_flux",
        minimum=15,
        maximum=1000,
    )
    mapper.make_parameter_map.assert_called_once_with(
        column_name="Noise",
        coordinate_system="equatorial",
        no_source_val=np.nan,
        nside=64,
        nest=False,
        operation="mean",
    )
    masker_cls.assert_called_once_with(parameter_map, coordinate_system="equatorial")
    masker.mask_galactic_plane.assert_called_once_with(latitude_cut=5)
    assert result is masked_map
    assert racs.parameter_maps["Noise"] is masked_map


def test_none_masks_are_skipped_and_non_none_masks_are_called() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    racs.override_defaults(
        masks={
            "mask_galactic_plane": {"kwargs": {"latitude_cut": 5}},
            "mask_equatorial_poles": None,
            "mask_slice": None,
        }
    )
    masker = MagicMock()
    racs._apply_masks(masker)

    masker.mask_galactic_plane.assert_called_once_with(latitude_cut=5)
    masker.mask_equatorial_poles.assert_not_called()
    masker.mask_slice.assert_not_called()


def test_mask_slice_supports_multiple_configured_calls() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    racs.override_defaults(
        masks={
            "mask_slice": [
                {"args": [0.0, 10.0, 3.0]},
                {"args": [20.0, -5.0, 4.0]},
            ]
        }
    )
    masker = MagicMock()
    racs._apply_masks(masker)

    masker.mask_slice.assert_any_call(0.0, 10.0, 3.0)
    masker.mask_slice.assert_any_call(20.0, -5.0, 4.0)
    assert masker.mask_slice.call_count == 2


def test_mask_slice_supports_shorthand_calls() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    racs.set_mask("mask_slice", [[0.0, 10.0, 3.0], [20.0, -5.0, 4.0]])
    masker = MagicMock()
    racs._apply_masks(masker)

    masker.mask_slice.assert_any_call(0.0, 10.0, 3.0)
    masker.mask_slice.assert_any_call(20.0, -5.0, 4.0)
    assert masker.mask_slice.call_count == 2


def test_override_defaults_is_instance_local() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        first = RACS("low3")
        second = RACS("low3")

    first.override_defaults(
        flux_min=20,
        masks={"mask_equatorial_poles": {"kwargs": {"north_radius": 42}}},
    )

    assert first.defaults["flux_min"] == 20
    assert second.defaults["flux_min"] == 15
    assert first.defaults["masks"]["mask_equatorial_poles"] == {
        "kwargs": {"north_radius": 42}
    }
    assert second.defaults["masks"]["mask_equatorial_poles"] != (
        first.defaults["masks"]["mask_equatorial_poles"]
    )


def test_named_helpers_update_defaults() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    racs.set_flux_cut(minimum=20, maximum=None)
    racs.set_equatorial_pole_mask(north_radius=50, south_radius=20)
    racs.disable_mask("mask_a_team_sources")

    assert racs.defaults["flux_min"] == 20
    assert racs.defaults["flux_max"] is None
    assert racs.defaults["masks"]["mask_equatorial_poles"] == {
        "north_radius": 50,
        "south_radius": 20,
    }
    assert racs.defaults["masks"]["mask_a_team_sources"] is None


def test_disable_catalogue_operation() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    racs.remove_local_sources(radius=5)
    racs.disable_catalogue_operation("crossmatch_local_sources")

    assert racs.defaults["catalogue_operations"]["crossmatch_local_sources"] is None


def test_describe_fiducial_map_shows_active_config() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    description = racs.describe_fiducial_map()

    assert "RACS variant: low3" in description
    assert "Flux cut: 15 <= Total_flux <= 1000" in description
    assert "Catalogue operations:" in description
    assert "mask_galactic_plane" in description
    assert "mask_ecliptic_poles" not in description


def test_invalid_override_raises_immediately() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    with pytest.raises(ValueError, match="Unknown RACS mask method"):
        racs.override_defaults(masks={"not_a_mask": {}})


def test_invalid_flux_column_override_raises_immediately() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    with pytest.raises(ValueError, match="flux column"):
        racs.override_defaults(flux_column="missing_flux")


def test_invalid_catalogue_operation_raises_immediately() -> None:
    with patch("dipoleutils.utils.racs.DataLoader") as data_loader_cls:
        data_loader_cls.return_value.load.return_value = _test_catalogue()
        racs = RACS("low3")

    with pytest.raises(ValueError, match="Unknown RACS catalogue operation"):
        racs.set_catalogue_operation("not_an_operation", {})
