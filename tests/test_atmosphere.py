import numpy as np
import pytest
import sarkit.wgs84
import scipy.constants

import sarkit_processing.atmosphere as atmo


def test_ellipsoid_refractivity():
    assert atmo.ellipsoid_refractivity(300.0, 100) > 300.0
    assert atmo.ellipsoid_refractivity(300.0, -100) < 300.0


def test_one_way_tropo_delay_to_space():
    assert atmo.one_way_tropo_delay_to_space(
        300.0, np.pi / 2
    ) < atmo.one_way_tropo_delay_to_space(300.0, np.pi / 4)
    assert atmo.one_way_tropo_delay_to_space(
        320.0, np.pi / 4
    ) > atmo.one_way_tropo_delay_to_space(300.0, np.pi / 4)
    assert 0 < atmo.one_way_tropo_delay_to_space(320.0, np.pi / 4) < 100e-9


def test_one_way_tropo_delay():
    ecef1 = sarkit.wgs84.geodetic_to_cartesian([0, 0, 0])
    ecef2 = sarkit.wgs84.geodetic_to_cartesian([0, 0, 1e5])
    assert atmo.one_way_tropo_delay(ecef1, ecef2, 0.0) == 0.0
    assert (
        0
        >= atmo.one_way_tropo_delay(ecef1, ecef2, 320.0)
        - atmo.one_way_tropo_delay_to_space(320.0, np.pi / 2)
        > -1e-9
    )


def test_iono_obliquity():
    ecefg = sarkit.wgs84.geodetic_to_cartesian([0, 0, 0])
    ecefa = sarkit.wgs84.geodetic_to_cartesian([5, 0, 600e3])

    assert 1.0 < atmo.iono_obliquity(ecefg, ecefa, 350e3) < 1.5
    assert 0.0 == atmo.iono_obliquity(ecefg, ecefa, 900e3)


def test_one_way_iono_coef():
    ecefg = sarkit.wgs84.geodetic_to_cartesian([0, 0, 0])
    ecefa = sarkit.wgs84.geodetic_to_cartesian([5, 0, 600e3])
    ecefaz = sarkit.wgs84.geodetic_to_cartesian([0, 0, 600e3])
    assert 0.0 == atmo.one_way_iono_coef(ecefg, ecefa, 900e3, 4.0)
    assert atmo.one_way_iono_coef(ecefg, ecefa, 350e3, 4.0) / atmo.one_way_iono_coef(
        ecefg, ecefa, 350e3, 2.0
    ) == pytest.approx(2.0)
    assert atmo.one_way_iono_coef(ecefg, ecefa, 350e3, 4.0) / atmo.one_way_iono_coef(
        ecefg, ecefaz, 350e3, 4.0
    ) == pytest.approx(atmo.iono_obliquity(ecefg, ecefa, 350e3))


def test_one_way_iono_delay():
    ecefg = sarkit.wgs84.geodetic_to_cartesian([0, 0, 0])
    ecefa = sarkit.wgs84.geodetic_to_cartesian(
        [0, 0, 20e6]
    )  # roughly GPS orbital height
    freq = 1.57542e9  # GPS L1

    # delay looked up with online GPS iono map is in meters at L1 for zenith
    assert (
        4.86
        < atmo.one_way_iono_delay(ecefg, ecefa, 350e3, 30.0, freq)
        * scipy.constants.speed_of_light
        < 4.88
    )
