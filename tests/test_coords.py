import itertools
import json
import shutil
import subprocess
import sys

import lxml.etree
import networkx as nx
import numpy as np
import pytest
import sarkit.sicd as sksicd
import sarkit.wgs84
import shapely
import shapely.geometry as shg

import sarkit_processing._geometry_utils as gu
import sarkit_processing.coords as skpc
import tests.utils
from sarkit_processing import _io


def test_transgeom():
    coords = np.arange(300).reshape(-1, 3)
    polygon = shg.Polygon(coords)

    def mytransform(x):
        return x + 1

    expected = shapely.transform(polygon, mytransform, include_z=True)

    new_polygon = skpc._transgeom(mytransform, include_z=True)(polygon)
    assert np.all(
        shapely.get_coordinates(new_polygon) == shapely.get_coordinates(expected)[:, :2]
    )

    new_polygon = skpc._transgeom(mytransform, include_z=True)(polygon)
    assert np.all(
        shapely.get_coordinates(new_polygon) == shapely.get_coordinates(expected)
    )

    with pytest.raises(ValueError, match="include Z"):
        skpc._transgeom(mytransform, include_z=False)(polygon)


def test_graph(example_sicd):

    def _get_path(this_graph, starting, ending):
        try:
            return nx.shortest_path(this_graph, starting, ending)
        except nx.NetworkXNoPath:
            return []

    basic_graph = skpc.create_conversion_graph()
    assert len(basic_graph.nodes) == 5
    assert len(basic_graph.edges) == 8
    assert skpc.ROWCOL not in basic_graph

    full_graph = skpc.create_conversion_graph(example_sicd)
    assert len(full_graph.nodes) == 8
    assert len(full_graph.edges) == 16
    assert skpc.ROWCOL in full_graph

    assert _get_path(basic_graph, gu.LATLONHAE, gu.LATLON)
    assert _get_path(basic_graph, gu.ECEF, gu.LONLAT)
    assert not _get_path(basic_graph, gu.LATLON, gu.LATLONHAE)
    assert _get_path(full_graph, gu.LATLONHAE, gu.LATLON)
    assert _get_path(full_graph, gu.LATLONHAE, skpc.ROWCOL)

    xmltree = _io.read_sicd_xml(example_sicd)
    ew = sksicd.ElementWrapper(xmltree.getroot())
    ecef_pt = ew["GeoData"]["SCP"]["ECF"]
    lat, lon, hae = sarkit.wgs84.cartesian_to_geodetic(ecef_pt)

    xryc, _, _ = sksicd.scene_to_image(xmltree, ecef_pt)
    iric = sksicd.xrowycol_to_irowicol(xmltree, xryc)
    rc = sksicd.xrowycol_to_rowcol(xmltree, xryc)

    geometries = {
        gu.LATLON: shg.Point([lat, lon]),
        gu.LONLAT: shg.Point([lon, lat]),
        gu.LATLONHAE: shg.Point([lat, lon, hae]),
        gu.LONLATHAE: shg.Point([lon, lat, hae]),
        gu.ECEF: shg.Point(ecef_pt),
        skpc.ROWCOL: shg.Point(rc),
        skpc.IROWICOL: shg.Point(iric),
        skpc.XROWYCOL: shg.Point(xryc),
    }

    path_lengths = {}
    for from_cs, to_cs in itertools.combinations(geometries.keys(), 2):
        path = _get_path(full_graph, from_cs, to_cs)
        path_lengths.setdefault(len(path), 0)
        if path:
            path_lengths[len(path)] += 1
            converted = skpc.convert(full_graph, from_cs, to_cs, geometries[from_cs])
            np.testing.assert_allclose(
                shapely.get_coordinates(converted),
                shapely.get_coordinates(geometries[to_cs]),
            )

    # regression test
    assert path_lengths == {
        2: 8,
        3: 7,
        4: 6,
        5: 5,
        6: 2,
    }

    graph = full_graph.copy()

    def _check_and_remove(from_cs, to_cs):
        converted = graph.edges[(from_cs, to_cs)]["handler"](geometries[from_cs])
        np.testing.assert_allclose(
            shapely.get_coordinates(converted),
            shapely.get_coordinates(geometries[to_cs]),
        )
        graph.remove_edge(from_cs, to_cs)

    # check every expected edge
    _check_and_remove(gu.LATLON, gu.LONLAT)
    _check_and_remove(gu.LONLAT, gu.LATLON)
    _check_and_remove(gu.LATLONHAE, gu.LONLATHAE)
    _check_and_remove(gu.LONLATHAE, gu.LATLONHAE)
    _check_and_remove(gu.LATLONHAE, gu.LATLON)
    _check_and_remove(gu.LONLATHAE, gu.LONLAT)
    _check_and_remove(gu.LATLONHAE, gu.ECEF)
    _check_and_remove(gu.ECEF, gu.LATLONHAE)
    _check_and_remove(gu.LATLON, gu.LATLONHAE)
    _check_and_remove(skpc.ROWCOL, skpc.IROWICOL)
    _check_and_remove(skpc.ROWCOL, skpc.XROWYCOL)
    _check_and_remove(skpc.IROWICOL, skpc.ROWCOL)
    _check_and_remove(skpc.IROWICOL, skpc.XROWYCOL)
    _check_and_remove(skpc.XROWYCOL, skpc.ROWCOL)
    _check_and_remove(skpc.XROWYCOL, skpc.IROWICOL)
    _check_and_remove(gu.ECEF, skpc.XROWYCOL)

    assert len(graph.edges) == 0


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
    polygon_geo = gu.swap_latlon(sarkit.wgs84.cartesian_to_geodetic(polygon_ecef))
    geojson_str = shapely.to_geojson(shg.Polygon(polygon_geo))

    # Coordinates via argument string
    proc = subprocess.run(
        [
            "coords",
            "--sicd",
            example_sicd,
            "--to-cs",
            "rowcol",
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
            "sarkit_processing.coords",
            "--sicd",
            xml_file,
            "--to-cs",
            "rowcol",
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
            "sarkit_processing.coords",
            "--sicd",
            example_sicd,
            "--to-cs",
            "rowcol",
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
            "sarkit_processing.coords",
            "--sicd",
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
            "sarkit_processing.coords",
            "--sicd",
            example_sicd,
            "--from-cs",
            "lonlathae",
            "--to-cs",
            "rowcol",
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
            "sarkit_processing.coords",
            "--sicd",
            example_sicd,
            "--from-cs",
            "ecef",
            "--to-cs",
            "rowcol",
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
            "sarkit_processing.coords",
            "--sicd",
            example_sicd,
            "--from-cs",
            "lonlathae",
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
            "sarkit_processing.coords",
            "--sicd",
            example_sicd,
            "--from-cs",
            "ecef",
            "--to-cs",
            "rowcol",
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
                "sarkit_processing.coords",
                "--sicd",
                f"{server_url}/{example_sicd.name}",
                "--to-cs",
                "rowcol",
                f"{server_url}/{geojson_file.name}",
            ],
            capture_output=True,
            check=True,
        )
        projected = shapely.from_wkt(proc.stdout)
        np.testing.assert_allclose(
            shapely.get_coordinates(projected)[0], ew["ImageData"]["SCPPixel"]
        )
