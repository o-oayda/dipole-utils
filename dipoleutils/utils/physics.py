import numpy as np
from numpy.typing import NDArray
from astropy.coordinates import (
    SkyCoord, ICRS, Galactic, BarycentricMeanEcliptic
)

def omega_to_theta(omega):
    '''
    Convert solid angle in steradins to theta in radians for a cone section
    of a sphere.
    
    :param omega: Solid angle in steradians.
    :return: Angle in radians,
    '''
    return np.arccos(1 - omega / (2 * np.pi))

def spherical_to_degrees(
        longitude_rad: NDArray[np.float64],
        colatitude_rad: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Convert spherical coordinates (phi ~ [0, 2pi], theta ~ [0, pi] starting from
    the north pole) to the same system but in degrees, with phi' ~ [0, 360] and
    theta' ~ [-90, 90]. In this new system, theta is 0 at the equator, 90 at the
    north pole and -90 at the south pole.

    :param longitude_rad: Longitude between 0 and 2pi.
    :param colatitude_rad: Colatitude between 0 and pi.
    :return: Tuple (phi', theta') of transformed coordinates.
    '''
    longitude_deg = np.rad2deg(longitude_rad)
    latitude_deg = np.rad2deg(np.pi / 2 - colatitude_rad)
    return longitude_deg, latitude_deg

def change_source_coordinates(
        source_longitude_deg: NDArray[np.float64],
        source_latitude_deg: NDArray[np.float64],
        native_coordinates: str,
        target_coordinates: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''
    Convert arrays of source longitudes and latitudes in one coordinate frame
    to another. Currently supported frames:

    - Equatorial ICRS
    - Galactic
    - Barycentric mean ecliptic

    :param source_longitude_deg: Array of source longitudes in degrees.
    :param source_latitude_deg: Array of source latitudes in degrees.
    :param native_coordinates: The coordinate frame in which the source
        longitudes and latitudes are specified.
    :param target_coordinates: The desired coordinate system to transform the
        sources to.
    :returns: Tuple containing transformed source longitudes and latitudes.
    :raises AssertionError: If unsupported/unrecognised frame is specified.
    :raises ValueError: If the coordinate transformation fails, resulting in
        a the transformed SkyCoord object being None.
    '''
    string_to_astropy_frame = {
        'equatorial': ICRS,
        'galactic': Galactic,
        'ecliptic': BarycentricMeanEcliptic
    }
    assert native_coordinates in string_to_astropy_frame.keys(), (
        f'Unrecognised native frame ({native_coordinates}) specified.'
    )
    assert target_coordinates in string_to_astropy_frame.keys(), (
        f'Unrecognised target frame ({target_coordinates}) specified.'
    )
    native_sources = SkyCoord(
        source_longitude_deg,
        source_latitude_deg,
        frame=string_to_astropy_frame[native_coordinates],
        unit='deg'
    )
    transformed_sources = native_sources.transform_to(
        frame=string_to_astropy_frame[target_coordinates]
    )
    if transformed_sources is None:
        raise ValueError("Transformation failed, resulting in None.")

    # Helper function to extract and convert attributes
    def extract_attributes(source, attr1, attr2):
        return (
            np.asarray(getattr(source, attr1).value, dtype=np.float64),
            np.asarray(getattr(source, attr2).value, dtype=np.float64)
        )

    # Map target coordinates to corresponding attributes
    frame_to_attributes = {
        'galactic': ('l', 'b'),
        'equatorial': ('ra', 'dec'),
        'ecliptic': ('lon', 'lat')
    }

    assert target_coordinates in frame_to_attributes.keys(), (
        f"Unsupported target_coordinates: {target_coordinates}"
    )

    attr1, attr2 = frame_to_attributes[target_coordinates]
    return extract_attributes(transformed_sources, attr1, attr2)

def compute_ellis_baldwin_amplitude(
        observer_speed: float | np.floating,
        luminosity_function_slope: float | np.floating,
        spectral_index: float | np.floating
) -> float | np.floating:
    '''
    Compute the Ellis & Baldwin (1984) amplitude expectation.

    :param observer_speed: Speed of the observer as a fraction of the speed of
        light. This is beta in the formula.
    :param luminosity_function_slope: Exponent parametrising the cumulative
        flux density distribution (integrated counts). This is x in the formula.
    :param spectral_index: Exponent parametrising the SED of sources. This is
        alpha in the formula.
    '''
    return (
        2 + luminosity_function_slope * (1 + spectral_index)
    ) * observer_speed