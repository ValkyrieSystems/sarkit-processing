import shutil

import numpy as np
import numpy.polynomial.polynomial as npp
import pytest
import sarkit.sicd as sksicd

import sarkit_processing.__main__
import sarkit_processing.sicd_alias as spsa
import tests.utils


@pytest.mark.parametrize("shift", [-2.5, -1.0, 0.0, 0.75, 3.0])
def test_shift_poly_axis(shift):
    rng = np.random.default_rng()
    coeffs = rng.normal(size=6)
    shifted = spsa._shift_poly_axis(coeffs, shift, 0)

    xvals = rng.normal(size=100)

    np.testing.assert_allclose(
        npp.polyval(xvals + shift, coeffs),
        npp.polyval(xvals, shifted),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_2d_shift_poly_axis(axis):
    rng = np.random.default_rng()
    coeffs = rng.normal(size=(5, 4))
    shift = 1.3

    shifted = spsa._shift_poly_axis(coeffs, shift, axis)

    xvals = rng.normal(size=50)
    yvals = rng.normal(size=50)

    if axis == 0:
        expected = npp.polyval2d(xvals + shift, yvals, coeffs)
    else:
        expected = npp.polyval2d(xvals, yvals + shift, coeffs)

    actual = npp.polyval2d(xvals, yvals, shifted)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "scale_x, scale_y",
    [
        (1.0, 1.0),
        (2.0, 3.0),
        (0.5, 4.0),
        (-2.0, 3.0),
        (-1.5, -0.5),
    ],
)
def test_polyscale2d(scale_x, scale_y):
    rng = np.random.default_rng()
    coeffs = rng.normal(size=(5, 4))
    scaled = spsa._polyscale2d(coeffs, scale_x, scale_y)

    xvals = rng.normal(size=100)
    yvals = rng.normal(size=100)

    expected = npp.polyval2d(scale_x * xvals, scale_y * yvals, coeffs)
    actual = npp.polyval2d(xvals, yvals, scaled)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_main(tmp_path, example_sicd_alias):
    output_file = tmp_path / "cleaned_example.sicd"
    sarkit_processing.__main__.main(
        [
            "sicd_alias",
            str(example_sicd_alias),
            str(output_file),
            "--threshold",
            "7.0",
            "--num-iters",
            "3",
            "--zones",
            "-1.0",
            "2.0",
            "--symmetric",
            "--dilate",
            "3",
        ],
    )

    assert output_file.is_file()


def test_sicd_alias_smart_open(tmp_path, example_sicd_alias):
    output_file = tmp_path / "out.sicd"

    shutil.copyfile(example_sicd_alias, tmp_path / example_sicd_alias.name)
    with tests.utils.static_http_server(tmp_path) as server_url:
        sarkit_processing.__main__.main(
            [
                "sicd_alias",
                f"{server_url}/{example_sicd_alias.name}",
                str(output_file),
            ],
        )

    assert output_file.exists()


def test_sicd_alias_thresh(example_sicd_alias):
    with (
        open(example_sicd_alias, "rb") as file,
        sksicd.NitfReader(file) as reader,
    ):
        xmltree = reader.metadata.xmltree
        image = reader.read_image()

    num_iters = 2
    zones = [-1.0, 1.0]
    _, removed_pwr_frac, removed_data_frac = spsa.prf_alias_removal(
        image.astype("complex64"), xmltree, num_iters, zones, threshold=8.0
    )

    _, removed_pwr_frac_1, removed_data_frac_1 = spsa.prf_alias_removal(
        image.astype("complex64"), xmltree, num_iters, zones, threshold=7.0
    )

    assert removed_pwr_frac < removed_pwr_frac_1
    assert removed_data_frac < removed_data_frac_1


def test_sicd_alias_dilate(example_sicd_alias):
    with (
        open(example_sicd_alias, "rb") as file,
        sksicd.NitfReader(file) as reader,
    ):
        xmltree = reader.metadata.xmltree
        image = reader.read_image()

    num_iters = 2
    zones = [-1.0, 1.0]
    _, removed_pwr_frac, removed_data_frac = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        num_iters,
        zones,
        threshold=7.0,
    )

    _, removed_pwr_frac_1, removed_data_frac_1 = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        num_iters,
        zones,
        threshold=7.0,
        dilate=3,
    )

    assert removed_pwr_frac < removed_pwr_frac_1
    assert removed_data_frac < removed_data_frac_1


def test_sicd_alias_iters(example_sicd_alias):
    with (
        open(example_sicd_alias, "rb") as file,
        sksicd.NitfReader(file) as reader,
    ):
        xmltree = reader.metadata.xmltree
        image = reader.read_image()

    zones = [-1.0, 1.0]
    _, removed_pwr_frac, removed_data_frac = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        2,
        zones,
        threshold=4.0,
    )

    _, removed_pwr_frac_1, removed_data_frac_1 = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        3,
        zones,
        threshold=4.0,
    )

    assert removed_pwr_frac_1 > removed_pwr_frac
    assert removed_data_frac_1 > removed_data_frac


def test_sicd_alias_prf(example_sicd_alias):
    with (
        open(example_sicd_alias, "rb") as file,
        sksicd.NitfReader(file) as reader,
    ):
        xmltree = reader.metadata.xmltree
        image = reader.read_image()

    sicdew = sksicd.ElementWrapper(xmltree.getroot())

    num_iters = 2
    zones = [-1.0, 1.0]
    _, removed_pwr_frac, removed_data_frac = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        num_iters,
        zones,
        threshold=4.0,
    )

    time_poly = sicdew["Grid"]["TimeCOAPoly"]
    ipp_poly = sicdew["Timeline"]["IPP"]["Set"][0]["IPPPoly"]
    coa = npp.polyval2d(0, 0, time_poly)
    prf_override = npp.polyval(coa, npp.polyder(ipp_poly))
    del sicdew["Timeline"]["IPP"]
    _, removed_pwr_frac_1, removed_data_frac_1 = spsa.prf_alias_removal(
        image.astype("complex64"),
        xmltree,
        num_iters,
        zones,
        threshold=4.0,
        prf_override=prf_override,
    )

    assert removed_pwr_frac == removed_pwr_frac_1
    assert removed_data_frac == removed_data_frac_1
