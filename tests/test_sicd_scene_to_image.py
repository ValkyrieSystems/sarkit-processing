import numpy as np
import sarkit.sicd as sksicd
import shapely
import shapely.geometry as shg

import sarkit_processing.sicd_scene_to_image as ssi


def test_shapely_scene_to_image(example_sicd):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree
        ew = sksicd.ElementWrapper(xmltree.getroot())

    pt_coord = (0, 0)
    line_coord = [(0, 0), (2, 2), (11, 22)]
    polygon_coord = [(0, 0), (10, 0), (10, 20), (0, 20), (0, 0)]

    hae0 = ew["GeoData"]["SCP"]["LLH"][2] + 1.5
    pt_ecef, _, _ = sksicd.image_to_constant_hae_surface(xmltree, pt_coord, hae0)
    line_ecef, _, _ = sksicd.image_to_constant_hae_surface(xmltree, line_coord, hae0)
    polygon_ecef, _, _ = sksicd.image_to_constant_hae_surface(
        xmltree, polygon_coord, hae0
    )

    pt_rt = ssi.shapely_scene_to_image(xmltree, shg.Point(pt_ecef))
    np.testing.assert_allclose(shapely.get_coordinates(pt_rt), [pt_coord], atol=0.001)

    line_rt = ssi.shapely_scene_to_image(xmltree, shg.LineString(line_ecef))
    np.testing.assert_allclose(shapely.get_coordinates(line_rt), line_coord, atol=0.001)

    polygon_rt = ssi.shapely_scene_to_image(xmltree, shg.Polygon(polygon_ecef))
    np.testing.assert_allclose(
        shapely.get_coordinates(polygon_rt), polygon_coord, atol=0.001
    )

    collection_rt = ssi.shapely_scene_to_image(
        xmltree,
        shg.GeometryCollection(
            [shg.Point(pt_ecef), shg.LineString(line_ecef), shg.Polygon(polygon_ecef)]
        ),
    )
    np.testing.assert_allclose(
        shapely.get_coordinates(geometry=collection_rt),
        [pt_coord] + line_coord + polygon_coord,
        atol=0.001,
    )
