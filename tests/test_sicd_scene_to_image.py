import json
import shutil
import subprocess
import sys

import lxml.etree
import numpy as np
import pytest
import sarkit.sicd as sksicd
import sarkit.wgs84
import shapely
import shapely.geometry as shg

import sarkit_processing.sicd_scene_to_image as ssi
import tests.utils


def _swap_latlon(ll):
    ll = np.array(ll)
    ll[..., [1, 0]] = ll[..., [0, 1]]
    return ll


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


def test_parse_coordinates():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0, 3.0]},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {"type": "Point", "coordinates": [1.0, 2.0, 3.0]},
                        {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    (0.0, 0.0),
                                    (0.1, 0.0),
                                    (0.1, 0.2),
                                    (0.0, 0.2),
                                    (0.0, 0.0),
                                ]
                            ],
                        },
                    ],
                },
            },
        ],
    }
    geometry, coord_sys = ssi._parse_coordinates(json.dumps(geojson))
    assert shapely.get_num_geometries(geometry) == 3
    assert shapely.get_num_geometries(geometry.geoms[-1]) == 2
    assert coord_sys == "lonlat"

    geo3d = shapely.force_3d(
        geometry, 1.1
    )  # shapely has trouble parsing WKT with mixed 2D & 3D
    geometry, coord_sys = ssi._parse_coordinates(geo3d.wkt)
    assert shapely.get_num_geometries(geometry) == 3
    assert shapely.get_num_geometries(geometry.geoms[-1]) == 2
    assert coord_sys is None

    geometry, coord_sys = ssi._parse_coordinates("1.1, 2.2")
    assert geometry.coords[:] == [(1.1, 2.2)]
    assert coord_sys is None

    geometry, coord_sys = ssi._parse_coordinates("1.1, 2.2, 3.3")
    assert geometry.coords[:] == [(1.1, 2.2, 3.3)]
    assert coord_sys is None

    with pytest.raises(RuntimeError, match="Failed to interpret"):
        ssi._parse_coordinates("foobar")


def test_cli(example_sicd, tmp_path):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree
        ew = sksicd.ElementWrapper(xmltree.getroot())
    xml_file = tmp_path / "sicd.xml"
    xml_file.write_bytes(lxml.etree.tostring(xmltree))

    polygon_coord = [(0, 0), (10, 0), (10, 20), (0, 20), (0, 0)]
    hae0 = ew["GeoData"]["SCP"]["LLH"][2] + 1.5
    polygon_ecef, _, _ = sksicd.image_to_constant_hae_surface(
        xmltree, polygon_coord, hae0
    )
    polygon_geo = _swap_latlon(sarkit.wgs84.cartesian_to_geodetic(polygon_ecef))
    geojson_str = shapely.to_geojson(shg.Polygon(polygon_geo))

    # Coordinates via argument string
    proc = subprocess.run(
        [
            "sicd_scene_to_image",
            example_sicd,
            geojson_str,
        ],
        capture_output=True,
        check=True,
    )
    projected_cli_string = shapely.from_wkt(proc.stdout)

    # Coordinates via stdin
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            xml_file,
            "-",
        ],
        input=geojson_str.encode(),
        capture_output=True,
        check=True,
    )
    projected_stdin = shapely.from_wkt(proc.stdout)

    # Coordinates via file
    geojson_file = tmp_path / "coords.geojson"
    geojson_file.write_text(geojson_str)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            geojson_file,
        ],
        capture_output=True,
        check=True,
    )
    projected_file = shapely.from_wkt(proc.stdout)
    assert projected_cli_string == projected_stdin
    assert projected_stdin == projected_file

    np.testing.assert_allclose(
        polygon_coord,
        sksicd.rowcol_to_xrowycol(
            xmltree, shapely.get_coordinates(projected_cli_string)
        ),
        atol=0.001,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            "--to-cs",
            "xrowycol",
            geojson_str,
        ],
        capture_output=True,
        check=True,
    )
    projected = shapely.from_wkt(proc.stdout)
    np.testing.assert_allclose(
        polygon_coord, shapely.get_coordinates(projected), atol=0.001
    )

    # WKT input
    wkt_str = shapely.to_wkt(shg.Polygon(polygon_geo), rounding_precision=-1)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            "--from-cs",
            "lonlat",
            wkt_str,
        ],
        capture_output=True,
        check=True,
    )
    projected_wkt_cli_string = shapely.from_wkt(proc.stdout)
    np.testing.assert_allclose(
        shapely.get_coordinates(projected_cli_string),
        shapely.get_coordinates(projected_wkt_cli_string),
    )

    # WKT ECEF input
    wkt_ecef_str = shapely.to_wkt(shg.Polygon(polygon_ecef), rounding_precision=-1)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            "--from-cs",
            "ecef",
            wkt_ecef_str,
        ],
        capture_output=True,
        check=True,
    )
    projected_wkt_ecef_cli_string = shapely.from_wkt(proc.stdout)
    np.testing.assert_allclose(
        shapely.get_coordinates(projected_wkt_ecef_cli_string),
        shapely.get_coordinates(projected_wkt_cli_string),
    )

    # CSV input
    csv_str = f"{polygon_geo[0, 0]}, {polygon_geo[0, 1]}, {polygon_geo[0, 2]}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            "--from-cs",
            "lonlat",
            "--to-cs",
            "xrowycol",
            csv_str,
        ],
        capture_output=True,
        check=True,
    )
    projected_csv_str = shapely.from_wkt(proc.stdout)
    np.testing.assert_allclose(
        shapely.get_coordinates(projected_csv_str)[0],
        polygon_coord[0],
        atol=0.001,
    )


def test_cli_empty(example_sicd):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sarkit_processing.sicd_scene_to_image",
            example_sicd,
            "--from-cs",
            "ecef",
            "0 0 0",
        ],
        capture_output=True,
        check=True,
    )
    assert proc.stdout.decode().strip() == "GEOMETRYCOLLECTION EMPTY"


def test_smart_open(tmp_path, example_sicd):
    with open(example_sicd, "rb") as file, sksicd.NitfReader(file) as reader:
        xmltree = reader.metadata.xmltree

    ew = sksicd.ElementWrapper(xmltree.getroot())

    geojson_file = tmp_path / "geo.json"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": (ew["GeoData"]["SCP"]["LLH"]).tolist(),
                },
            },
        ],
    }
    geojson_file.write_text(json.dumps(geojson))

    shutil.copyfile(example_sicd, tmp_path / example_sicd.name)

    with tests.utils.static_http_server(tmp_path) as server_url:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sarkit_processing.sicd_scene_to_image",
                f"{server_url}/{example_sicd.name}",
                f"{server_url}/{geojson_file.name}",
            ],
            capture_output=True,
            check=True,
        )
        projected = shapely.from_wkt(proc.stdout)
        np.testing.assert_allclose(
            shapely.get_coordinates(projected)[0], ew["ImageData"]["SCPPixel"]
        )
