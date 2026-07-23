import numpy as np
import scipy.constants
from sarkit import wgs84

METERS_PER_FOOT = 0.3048

# Global average values from ITU-R P.453-11 (07/2015: equation 12)
TROPO_H0 = 7350.0
DEFAULT_N0 = 315.0


def ellipsoid_refractivity(tropo_ns, hae):
    """Compute the modeled refractivity at the ellispoid given the refractivity at the surface at a given height.

    Parameters
    ----------
    tropo_ns: float
        Troposphere refractivity at surface
    hae : float
        Surface height above the ellipsoid (m)

    Returns
    -------
    n_0 : float
        Troposphere refractivity at HAE=0
    """

    return tropo_ns * np.exp(hae / TROPO_H0)


def one_way_tropo_delay_to_space(ns, graze):
    """Compute one-way time delay due to troposphere from a point on the surface to space.

    This is a simplified version of the Altshuler [1]_ model which is an
    approximate empirical relationship that is moderately accurate over
    reasonable ranges of ellipsoid-level refractivity and grazing angles.

    By convention:
        Index of refraction is defined as ``n = c/v``. (Always >=  1.0)
        Refractivity is defined as ``N = (n-1)*1E06``

    Parameters
    ----------
    ns : array_like
        Surface refractivity
    graze : array_like
        Grazing angle in radians

    Returns
    -------
    ndarray
        One-way time delay in seconds

    References
    ----------
    .. [1] E. Altshuler, "Corrections for Tropospheric Range Error," AFCRL-71-0419, 1971

    """
    ns, graze = np.broadcast_arrays(ns, graze)
    # Section 3 equation (5)
    return (
        np.where(
            ns == 0,
            0.0,
            (4.79 + 0.00972 * ns) / np.sin(graze)
            - (0.00586 * (ns - 360) ** 2 + 294) * np.degrees(graze) ** -2.30,
        )
        * METERS_PER_FOOT
        / scipy.constants.speed_of_light
    )


def one_way_tropo_delay(gpt, apc, tropo_n0):
    """Compute one-way time delay due to troposphere.

    Returns the additional time (compared to free-space travel time) that a pulse will take
    to travel between the two input points due to tropospheric refractivity.

    Uses `one_way_tropo_delay_to_space` for much of the computation with a simple exponential model for correcting for
    the elevated point not being above the entire atmosphere.

    Parameters
    ----------
    gpt : array_like, shape=(...,3)
        Contains ECEF coordinates in meters (x,y,z) of ground point end of the path to be computed.

    apc : array_like, shape=(...,3)
        Contains ECEF coordinates in meters (x,y,z) of the elevated end of the path to be computed.

    tropo_n0 : float
        Refractivity (N) of the troposphere at HAE=0.  A special case is `tropo_n0 == 0` in which case the delay is 0.

    Returns
    -------
    float
        Additional time above the free-space travel time required for light to
        travel between the two points (in seconds).

    References
    ----------
    .. [1] E. Altshuler, "Corrections for Tropospheric Range Error," AFCRL-71-0419, 1971


    """
    gpt, apc = np.broadcast_arrays(gpt, apc)

    gpt_llh = wgs84.cartesian_to_geodetic(gpt)
    apc_llh = wgs84.cartesian_to_geodetic(apc)
    up = wgs84.up(gpt_llh)
    los_gpt_to_apc = apc - gpt
    sin_graze_ang = np.vecdot(
        los_gpt_to_apc / np.linalg.norm(los_gpt_to_apc, axis=-1, keepdims=True), up
    )
    graze = np.arcsin(sin_graze_ang)
    ns = tropo_n0 * np.exp(-gpt_llh[..., 2] / TROPO_H0)
    delay_gpt_to_space = one_way_tropo_delay_to_space(ns, graze)

    ha = apc_llh[..., 2] - gpt_llh[..., 2]
    ha_kft = ha / (METERS_PER_FOOT * 1e3)
    # Section 3 equation 6
    with np.errstate(divide="ignore", invalid="ignore"):
        delay_apc_to_space = delay_gpt_to_space * np.exp(
            -((6.07e-5 * ns + 0.0213) * ha_kft + (0.077 / ns - 1.58e-4) * ha_kft**2)
        )
    return np.where(ns == 0, 0.0, delay_gpt_to_space - delay_apc_to_space)


def iono_obliquity(gpt, apc, f2_height):
    """
    Computes the obliquity scaling for converting vertically integrated TEC to path integrated.

    Parameters
    ----------
    gpt: array_like
        ECEF position of the point near the ground
    apc: array_like
       ECEF position of the point in space
    f2_height: array_like
        Modeled height of the ionospheric layer

    Returns
    -------
    ndarray
        Obliquity factor that maps vertical delay to slant delay

    Notes
    -----
    This function approximates the ionosphere as an infinitely thin spherical
    shell. This means that there will be a discontinuity if the apc crosses the
    shell.


    This also assumes that `gpt` is inside the shell.

    """
    los = gpt - apc
    r_los = np.linalg.norm(los, axis=-1)
    r_apc = np.linalg.norm(apc, axis=-1)
    cos_alpha = np.vecdot(los, -apc) / (r_los * r_apc)
    f2_radius = f2_height + wgs84.SEMI_MAJOR_AXIS

    den = (f2_radius**2 - r_apc**2 * (1 - cos_alpha**2)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r_apc > f2_radius, f2_radius / den, 0.0)


def one_way_iono_coef(gpt, apc, f2_height, tec_v):
    """Calculate the coefficient on the size of the ionospheric delay vs frequency.

    The ionospheric delay is modeled as ``one_way_iono_coef/f**2`` [1]_.

    Parameters
    ----------
    gpt: array_like
        ECEF position of the point near the ground
    apc: array_like
       ECEF position of the point in space
    f2_height: array_like
        Modeled height of the ionospheric layer
    tec_v: array_like
        Total electron content integrated vertically from the ground in TECU (10**16 electrons / m**2)

    Returns
    -------
    ndarray
        model coefficient


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_electron_content#Propagation_delay
    """
    tec = tec_v * 1e16 * iono_obliquity(gpt, apc, f2_height)
    kappa = (
        scipy.constants.elementary_charge**2
        / (scipy.constants.electron_mass * scipy.constants.epsilon_0)
        / (8 * np.pi**2)
    )
    return tec * kappa / scipy.constants.speed_of_light


def one_way_iono_delay(gpt, apc, f2_height, tec_v, frequency):
    """Calculate the one-way delay due to ionosphere.

    Ignores path-bending effects of the ionosphere.

    Parameters
    ----------
    gpt: array_like
        ECEF position of the point near the ground
    apc: array_like
       ECEF position of the point in space
    f2_height: array_like
        Modeled height of the ionospheric layer
    tec_v: array_like
        Total electron content integrated vertically from the ground in TECU (10**16 electrons / m**2)
    frequency: array_like
        Center frequency for which to calculate group delay

    Returns
    -------
    ndarray
        delay in seconds

    See Also
    --------
    one_way_iono_coef : model used for the ionosphere
    """
    return one_way_iono_coef(gpt, apc, f2_height, tec_v) / frequency**2
