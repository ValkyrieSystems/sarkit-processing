import numpy as np
import pytest
import sarkit.sicd as sksicd
import sarkit.verification
import shapely.geometry as shg

import sarkit_processing.sicd_area_plane as sap


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((4, 1, 0), (8, 2)),
        ((4, 1, 0.5 * np.pi), (2, 8)),
        ((4, 1, np.pi), (8, 2)),
        ((4, 1, 1.5 * np.pi), (2, 8)),
        ((4, 1, 2 * np.pi), (8, 2)),
        ((4, 1, -0.5 * np.pi), (2, 8)),
        ((4, 0, 0), (8, 0)),
        ((4, 0, 0.5 * np.pi), (0, 8)),
    ],
)
def test_ellipse_axis_extents_cardinal(test_input, expected):
    extents = sap._ellipse_axis_extents(test_input[0], test_input[1], test_input[2])
    assert pytest.approx(extents) == np.asarray(expected)


@pytest.mark.parametrize("angle", np.pi * np.arange(-0.25, 2.25, 0.5))
def test_ellipse_axis_extents_diagonals(angle):
    axis_lengths = np.random.default_rng().random(2)
    extents = sap._ellipse_axis_extents(axis_lengths[0], axis_lengths[1], angle)
    assert pytest.approx(extents[0]) == extents[1]


@pytest.mark.parametrize(
    "test_input", [(4, 1, 0.75), (1, 4, 0.80), (1, 2, 2.35), (3, 1, 2.40)]
)
def test_ellipse_axis_extents_relative(test_input):
    extents = sap._ellipse_axis_extents(test_input[0], test_input[1], test_input[2])
    assert extents[0] > extents[1]


def test_area_plane_smoke(example_sicd):
    with open(example_sicd, "rb") as fd, sarkit.sicd.NitfReader(fd) as reader:
        sicd_xmltree = reader.metadata.xmltree
    area = sicd_xmltree.find("./{*}RadarCollection/{*}Area")
    if area is not None:
        area.getparent().remove(area)

    sicd_xmltree = sap.ensure_radarcollection_area_plane(sicd_xmltree)
    plane = sicd_xmltree.find("./{*}RadarCollection/{*}Area/{*}Plane")
    assert plane is not None

    con = sarkit.verification.SicdConsistency(sicd_xmltree.getroot())
    con.check()
    assert not con.failures()


def test_recompute_area_plane(example_sicd):
    with open(example_sicd, "rb") as fd, sarkit.sicd.NitfReader(fd) as reader:
        sicd_xmltree = reader.metadata.xmltree

    area = sicd_xmltree.find("./{*}RadarCollection/{*}Area")
    area.getparent().remove(area)

    new_sicd_xmltree = sap.recompute_area_plane(sicd_xmltree)
    assert (
        float(
            new_sicd_xmltree.find(
                "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}Line"
            ).text
        )
        != 0
    )
    new_sicd_xmltree.find(
        "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}Line"
    ).text = "0"

    # plane is overwritten if it already exists
    newnew_sicd_xmltree = sap.recompute_area_plane(new_sicd_xmltree)
    assert (
        float(
            newnew_sicd_xmltree.find(
                "./{*}RadarCollection/{*}Area/{*}Plane/{*}RefPt/{*}Line"
            ).text
        )
        != 0
    )


def test_create_area_node(example_sicd):
    with open(example_sicd, "rb") as fd, sarkit.sicd.NitfReader(fd) as reader:
        sicd_xmltree = reader.metadata.xmltree
    area = sicd_xmltree.find("./{*}RadarCollection/{*}Area")
    area.getparent().remove(area)

    plane = sap.compute_suitable_rc_plane(sicd_xmltree)

    area = sap.create_area_node(plane)
    assert area.find("./{*}Corner") is not None


def test_suitable_rc_plane(example_sicd):
    with open(example_sicd, "rb") as fd, sarkit.sicd.NitfReader(fd) as reader:
        sicd_xmltree = reader.metadata.xmltree
        sicd_ew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    area = sicd_xmltree.find("./{*}RadarCollection/{*}Area")
    area.getparent().remove(area)

    plane = sap.compute_suitable_rc_plane(sicd_xmltree)

    num_lines = int(plane.findtext("./{*}XDir/{*}NumLines"))
    line_spacing = float(plane.findtext("./{*}XDir/{*}LineSpacing"))
    x_uvect = sksicd.XyzType().parse_elem(plane.find("./{*}XDir/{*}UVectECF"))

    num_samples = int(plane.findtext("./{*}YDir/{*}NumSamples"))
    sample_spacing = float(plane.findtext("./{*}YDir/{*}SampleSpacing"))
    y_uvect = sksicd.XyzType().parse_elem(plane.find("./{*}YDir/{*}UVectECF"))

    ref_pt_ecf = sksicd.XyzType().parse_elem(plane.find("./{*}RefPt/{*}ECF"))
    ref_pt_line = float(plane.findtext("./{*}RefPt/{*}Line"))
    ref_pt_sample = float(plane.findtext("./{*}RefPt/{*}Sample"))

    basis = np.asarray([x_uvect, y_uvect])

    # Axes are orthogonal
    assert np.dot(x_uvect, y_uvect) == pytest.approx(0)

    scp_lat, scp_lon = np.deg2rad(sicd_ew["GeoData"]["SCP"]["LLH"][:2])
    up = [
        np.cos(scp_lat) * np.cos(scp_lon),
        np.cos(scp_lat) * np.sin(scp_lon),
        np.sin(scp_lat),
    ]
    up /= np.linalg.norm(up)

    # Plane is ETP as SCP
    np.testing.assert_almost_equal(np.cross(x_uvect, y_uvect), up)

    plane_perimeter = shg.Polygon(
        [
            (0.0, 0.0),
            (0.0, num_samples - 1),
            (num_lines - 1, num_samples - 1),
            (num_lines - 1, 0.0),
        ]
    ).segmentize(50)

    ref_pt_ls = np.asarray([ref_pt_line, ref_pt_sample])
    ss = np.asarray([line_spacing, sample_spacing])

    coords_xy = (plane_perimeter.exterior.coords - ref_pt_ls) * ss
    coords_ecf = ref_pt_ecf + coords_xy @ basis
    plane_perimeter_xy, _, _ = sksicd.scene_to_image(sicd_xmltree, coords_ecf)
    input_ss = np.asarray([sicd_ew["Grid"]["Row"]["SS"], sicd_ew["Grid"]["Col"]["SS"]])
    scp_rc = sicd_ew["ImageData"]["SCPPixel"]
    first_rc = np.asarray(
        [sicd_ew["ImageData"]["FirstRow"], sicd_ew["ImageData"]["FirstCol"]]
    )
    plane_perimeter_in_sicd_rc = plane_perimeter_xy / input_ss + scp_rc
    plane_perimeter_in_sicd_rc += first_rc  # Handle sub-images

    sicd_num_rows = sicd_ew["ImageData"]["NumRows"]
    sicd_num_cols = sicd_ew["ImageData"]["NumCols"]
    sicd_perimeter = shg.Polygon(
        [
            [0, 0],
            [0, sicd_num_cols - 1],
            [sicd_num_rows - 1, sicd_num_cols - 1],
            [sicd_num_rows - 1, 0],
        ]
    )

    valid_rc_poly = sicd_ew["ImageData"]["ValidData"]
    if valid_rc_poly is not None:
        sicd_rc_poly = sicd_perimeter.intersection(shg.Polygon(valid_rc_poly))
    else:
        sicd_rc_poly = sicd_perimeter

    # Area Plane should contain valid region of the SICD
    assert (
        shg.Polygon(plane_perimeter_in_sicd_rc).intersection(sicd_rc_poly).area
        > 0.99 * sicd_rc_poly.area
    )
