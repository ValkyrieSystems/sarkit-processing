import copy

import lxml.builder
import lxml.etree
import numpy as np
import sarkit.sicd
import sarkit.sicd.projection
import sarkit.wgs84
import shapely
import shapely.geometry as shg


def _element_maker(node):
    return lxml.builder.ElementMaker(
        namespace=lxml.etree.QName(node).namespace, nsmap=node.nsmap
    )


def ensure_radarcollection_area_plane(sicd_xmltree):
    """Make sure a SICD XML contains a RadarCollection/Area/Plane node"""
    if sicd_xmltree.find("./{*}RadarCollection/{*}Area/{*}Plane") is not None:
        return sicd_xmltree
    return recompute_area_plane(sicd_xmltree)


def recompute_area_plane(sicd_xmltree):
    """Populate the RadarCollection/Area/Plane node"""
    sicd_xmltree = copy.deepcopy(sicd_xmltree)
    plane = compute_suitable_rc_plane(sicd_xmltree)
    area = create_area_node(plane)

    radar_collection = sicd_xmltree.find("./{*}RadarCollection")
    old_area = radar_collection.find("{*}Area")
    if old_area is not None:
        radar_collection.remove(old_area)

    rcv_chan = radar_collection.find("./{*}RcvChannels")
    radar_collection.insert(radar_collection.index(rcv_chan) + 1, area)

    return sicd_xmltree


def create_area_node(plane):
    """Create a RadarCollection/Area node"""
    shape = np.asarray(
        (
            int(plane.findtext("{*}XDir/{*}NumLines")),
            int(plane.findtext("{*}YDir/{*}NumSamples")),
        )
    )
    spacing = np.asarray(
        (
            float(plane.findtext("{*}XDir/{*}LineSpacing")),
            float(plane.findtext("{*}YDir/{*}SampleSpacing")),
        )
    )
    ref_pt_ecef = np.asarray(
        (
            float(plane.findtext("./{*}RefPt/{*}ECF/{*}X")),
            float(plane.findtext("./{*}RefPt/{*}ECF/{*}Y")),
            float(plane.findtext("./{*}RefPt/{*}ECF/{*}Z")),
        )
    )
    ref_pt_ls = np.asarray(
        (
            float(plane.findtext("./{*}RefPt/{*}Line")),
            float(plane.findtext("./{*}RefPt/{*}Sample")),
        )
    )
    first = np.asarray(
        (
            float(plane.findtext("{*}XDir/{*}FirstLine")),
            float(plane.findtext("{*}YDir/{*}FirstSample")),
        )
    )
    x_uvect = np.asarray(
        (
            float(plane.findtext("./{*}XDir/{*}UVectECF/{*}X")),
            float(plane.findtext("./{*}XDir/{*}UVectECF/{*}Y")),
            float(plane.findtext("./{*}XDir/{*}UVectECF/{*}Z")),
        )
    )
    y_uvect = np.asarray(
        (
            float(plane.findtext("./{*}YDir/{*}UVectECF/{*}X")),
            float(plane.findtext("./{*}YDir/{*}UVectECF/{*}Y")),
            float(plane.findtext("./{*}YDir/{*}UVectECF/{*}Z")),
        )
    )
    basis = np.asarray([x_uvect, y_uvect])

    last = first + shape
    corners_ls = np.asarray(
        [
            (first[0], first[1]),
            (first[0], last[1]),
            (last[0], last[1]),
            (last[0], first[1]),
        ]
    )
    corners_xy = (corners_ls - ref_pt_ls) * spacing
    corners_ecef = ref_pt_ecef + corners_xy @ basis
    corners_llh = sarkit.wgs84.cartesian_to_geodetic(corners_ecef)
    sicdem = _element_maker(plane)
    area = sicdem.Area(
        sicdem.Corner(
            *[
                sicdem.ACP(
                    sicdem.Lat(str(corners_llh[idx][0])),
                    sicdem.Lon(str(corners_llh[idx][1])),
                    sicdem.HAE(str(corners_llh[idx][2])),
                    index=str(idx + 1),
                )
                for idx in range(4)
            ]
        ),
        plane,
    )
    return area


def _ellipse_axis_extents(semi_major, semi_minor, angle):
    """Computes the length of the line of intersection between the x and y axes
    and an ellipse rotated in the x-y plane.

    Parameters
    ----------
    semi_major : float
        The length of semi-major axis.
    semi_minor : float
        The length of semi-minor axis.
    angle : float
        The angle between the plane's x axis and the semi-major axis of the ellipse.

    Returns
    -------
    axis_intersection_lengths : `numpy.ndarray`
        The lengths of intersection between the ellipse and the x and y axes respectively.
    """
    if np.any(np.isclose(angle, [0, np.pi])):
        return 2 * np.asarray([semi_major, semi_minor])
    if np.any(np.isclose(angle, [0.5 * np.pi, 1.5 * np.pi])):
        return 2 * np.asarray([semi_minor, semi_major])
    cos = np.cos(angle)
    sin = np.sin(angle)
    x_intersect_length = 2 * (
        semi_major
        * semi_minor
        / np.sqrt(semi_major**2 * sin**2 + semi_minor**2 * cos**2)
    )
    y_intersect_length = 2 * (
        semi_major
        * semi_minor
        / np.sqrt(semi_major**2 * cos**2 + semi_minor**2 * sin**2)
    )
    return np.asarray([x_intersect_length, y_intersect_length])


def compute_suitable_rc_plane(sicd_xmltree):
    """Compute a RadarCollection/Area/Plane node"""
    xml_helper = sarkit.sicd.XmlHelper(sicd_xmltree)

    delta_xrow = 1.0 / xml_helper.load("./{*}Grid/{*}Row/{*}ImpRespBW")
    delta_ycol = 1.0 / xml_helper.load("./{*}Grid/{*}Col/{*}ImpRespBW")
    proj_meta = sarkit.sicd.projection.MetadataParams.from_xml(sicd_xmltree)
    mats = sarkit.sicd.projection.compute_sensitivity_matrices(
        proj_meta, delta_xrow=delta_xrow, delta_ycol=delta_ycol
    )
    ground_resolution_proj = mats.M_GPXY_IL * np.array([delta_xrow, delta_ycol])

    eigen_val, eigen_vec = np.linalg.eig(
        ground_resolution_proj @ ground_resolution_proj.T
    )
    max_val = np.argmax(eigen_val)

    scene_coord_angle = np.arctan2(eigen_vec[max_val][1], eigen_vec[max_val][0])
    semi_major = np.sqrt(np.abs(eigen_val[max_val]))
    semi_minor = np.sqrt(np.abs(eigen_val[(max_val + 1) % 2]))

    required_sampling = (
        _ellipse_axis_extents(semi_major, semi_minor, scene_coord_angle) / 2
    )

    sample_spacing = 0.886 * min(required_sampling) / 1.5
    output_ss = (sample_spacing, sample_spacing)

    scp_ecef = xml_helper.load("./{*}GeoData/{*}SCP/{*}ECF")
    scp_pixel = xml_helper.load("./{*}ImageData/{*}SCPPixel")
    arppos_ecef = xml_helper.load("./{*}SCPCOA/{*}ARPPos")

    def _unit(vec):
        return vec / np.linalg.norm(vec, axis=-1)

    sp_x = _unit(scp_ecef - arppos_ecef)
    gp_z = _unit(sarkit.wgs84.up(xml_helper.load("./{*}GeoData/{*}SCP/{*}LLH")))
    gp_y = _unit(np.cross(gp_z, sp_x))
    gp_x = _unit(np.cross(gp_y, gp_z))

    sicd_ss = (
        (xml_helper.load("./{*}Grid/{*}Row/{*}SS")),
        (xml_helper.load("./{*}Grid/{*}Col/{*}SS")),
    )
    sicd_num_rows = xml_helper.load("./{*}ImageData/{*}NumRows")
    sicd_num_cols = xml_helper.load("./{*}ImageData/{*}NumCols")
    extent_rc_poly = shg.Polygon(
        [
            [0, 0],
            [0, sicd_num_cols - 1],
            [sicd_num_rows - 1, sicd_num_cols - 1],
            [sicd_num_rows - 1, 0],
        ]
    )

    # TODO create a "get_valid_data" function
    valid_rc_poly = xml_helper.load("./{*}ImageData/{*}ValidData")
    if valid_rc_poly is not None:
        sicd_rc_poly = extent_rc_poly.intersection(shg.Polygon(valid_rc_poly))
    else:
        sicd_rc_poly = extent_rc_poly
    sicd_xy_poly = shg.Polygon(
        (np.array(sicd_rc_poly.exterior.coords) - scp_pixel) * sicd_ss
    )

    max_edge_length = sicd_xy_poly.length / 40
    dense_perimeter = shapely.get_coordinates(sicd_xy_poly.segmentize(max_edge_length))

    initial_output_coords_ecef, _, _ = sarkit.sicd.image_to_ground_plane(
        sicd_xmltree, dense_perimeter, gref=scp_ecef, ugpn=gp_z
    )
    initial_output_coords = np.dot(
        initial_output_coords_ecef - scp_ecef, np.asarray([gp_x, gp_y]).T
    )
    mrr = shg.Polygon(initial_output_coords).minimum_rotated_rectangle

    mrr_coords = np.array(mrr.exterior.coords)
    axis = mrr_coords[3] - mrr_coords[0]
    rot_angle = np.arctan2(axis[1], axis[0])
    rot_angle = (rot_angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
    new_gp_x = np.cos(rot_angle) * gp_x + np.sin(rot_angle) * gp_y
    new_gp_y = -np.sin(rot_angle) * gp_x + np.cos(rot_angle) * gp_y

    output_coords = np.dot(
        initial_output_coords_ecef - scp_ecef, np.asarray([new_gp_x, new_gp_y]).T
    )

    scene_shape_float = np.ptp(output_coords, axis=0) / output_ss
    scene_shape = np.floor(scene_shape_float).astype(np.uint64)
    frac = (scene_shape - scene_shape_float) / 2.0
    ref_pt_ls = -output_coords.min(axis=0) / output_ss + frac

    sicdem = lxml.builder.ElementMaker(
        namespace=lxml.etree.QName(sicd_xmltree.getroot()).namespace,
        nsmap=sicd_xmltree.getroot().nsmap,
    )
    plane = sicdem.Plane(
        sicdem.RefPt(
            sicdem.ECF(
                sicdem.X(str(scp_ecef[0])),
                sicdem.Y(str(scp_ecef[1])),
                sicdem.Z(str(scp_ecef[2])),
            ),
            sicdem.Line(str(ref_pt_ls[0])),
            sicdem.Sample(str(ref_pt_ls[1])),
        ),
        sicdem.XDir(
            sicdem.UVectECF(
                sicdem.X(str(new_gp_x[0])),
                sicdem.Y(str(new_gp_x[1])),
                sicdem.Z(str(new_gp_x[2])),
            ),
            sicdem.LineSpacing(str(output_ss[0])),
            sicdem.NumLines(str(scene_shape[0])),
            sicdem.FirstLine("0"),
        ),
        sicdem.YDir(
            sicdem.UVectECF(
                sicdem.X(str(new_gp_y[0])),
                sicdem.Y(str(new_gp_y[1])),
                sicdem.Z(str(new_gp_y[2])),
            ),
            sicdem.SampleSpacing(str(output_ss[1])),
            sicdem.NumSamples(str(scene_shape[1])),
            sicdem.FirstSample("0"),
        ),
    )
    return plane
