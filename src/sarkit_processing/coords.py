import argparse
import functools
import itertools

import networkx as nx
import sarkit.sicd as sksicd
import sarkit.wgs84
import shapely

import sarkit_processing._geometry_utils as gu
import sarkit_processing.sicd_scene_to_image as ss2i
from sarkit_processing import _io

ROWCOL = "rowcol"
IROWICOL = "irowicol"
XROWYCOL = "xrowycol"


def _transgeom(coordinate_transform, include_z):
    """Create a callable that applies a coordinate transform to a geometry"""

    def innter(geometry):
        # Be more strict than shapely and error out instead of introducing NaNs
        if include_z != geometry.has_z:
            raise ValueError("include Z mismatch")

        return shapely.transform(geometry, coordinate_transform, include_z=include_z)

    return innter


def create_conversion_graph(sicdfile=None):
    """Directional graph of coordinate system conversions

    Parameters
    ----------
    sicdfile : filename, optional
        Path to a SICD NITF or XML file.  When provided, conversions requiring a SICD are included in the graph.

    Returns
    -------
    networkx.DiGraph
        Directional graph where each node represents a coordinate system.
        Edges between nodes will have a "handler" attribute which is a function that takes a shapely geometry and returns a shapely geometry.
    """
    graph = nx.DiGraph()

    graph.add_edge(
        gu.LATLON,
        gu.LONLAT,
        handler=_transgeom(gu.swap_latlon, include_z=False),
    )
    graph.add_edge(
        gu.LONLAT,
        gu.LATLON,
        handler=_transgeom(gu.swap_latlon, include_z=False),
    )
    graph.add_edge(
        gu.LATLONHAE,
        gu.LONLATHAE,
        handler=_transgeom(gu.swap_latlon, include_z=True),
    )
    graph.add_edge(
        gu.LONLATHAE,
        gu.LATLONHAE,
        handler=_transgeom(gu.swap_latlon, include_z=True),
    )
    graph.add_edge(gu.LATLONHAE, gu.LATLON, handler=shapely.force_2d)
    graph.add_edge(gu.LONLATHAE, gu.LONLAT, handler=shapely.force_2d)
    graph.add_edge(
        gu.LATLONHAE,
        gu.ECEF,
        handler=_transgeom(sarkit.wgs84.geodetic_to_cartesian, include_z=True),
    )
    graph.add_edge(
        gu.ECEF,
        gu.LATLONHAE,
        handler=_transgeom(sarkit.wgs84.cartesian_to_geodetic, include_z=True),
    )

    if sicdfile is not None:
        xmltree = _io.read_sicd_xml(sicdfile)
        scp_hae = sksicd.XmlHelper(xmltree).load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")
        graph.add_edge(
            gu.LATLON,
            gu.LATLONHAE,
            handler=lambda geom: shapely.force_3d(geom, scp_hae),
        )

        graph.add_edge(
            ROWCOL,
            IROWICOL,
            handler=_transgeom(
                functools.partial(sksicd.rowcol_to_irowicol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            ROWCOL,
            XROWYCOL,
            handler=_transgeom(
                functools.partial(sksicd.rowcol_to_xrowycol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            IROWICOL,
            ROWCOL,
            handler=_transgeom(
                functools.partial(sksicd.irowicol_to_rowcol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            IROWICOL,
            XROWYCOL,
            handler=_transgeom(
                functools.partial(sksicd.irowicol_to_xrowycol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            XROWYCOL,
            ROWCOL,
            handler=_transgeom(
                functools.partial(sksicd.xrowycol_to_rowcol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            XROWYCOL,
            IROWICOL,
            handler=_transgeom(
                functools.partial(sksicd.xrowycol_to_irowicol, xmltree), include_z=False
            ),
        )
        graph.add_edge(
            gu.ECEF,
            XROWYCOL,
            handler=functools.partial(ss2i.shapely_scene_to_image, xmltree),
        )

    return graph


def convert(graph, from_cs, to_cs, geometry):
    """Convert a geometry from one coordinate system to another

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph of coordinate system conversions.  See: `create_conversion_graph`
    from_cs : str
        Coordinate system of provided geometry
    to_cs : str
        Coordinate system to convert to
    geometry : shapely.Geometry
        geometry to be converted

    Returns
    -------
    shapely.Geometry
        geometry with converted coordinates
    """
    if from_cs != to_cs:
        path = nx.shortest_path(graph, from_cs, to_cs)

        for src, dest in itertools.pairwise(path):
            geometry = graph.edges[(src, dest)]["handler"](geometry)

    return geometry


def main(args=None):
    coordinate_systems = {
        gu.LATLON: "2D WGS84 geodetic [latitude (deg), longitude (deg)]",
        gu.LONLAT: "2D WGS84 geodetic [longitude (deg), latitude (deg)]",
        gu.LATLONHAE: "3D WGS84 geodetic [latitude (deg), longitude (deg), height (m)]",
        gu.LONLATHAE: "3D WGS84 geodetic [longitude (deg), latitude (deg), height (m)]",
        gu.ECEF: "3D WGS84 cartesian [X (m), Y (m), Z (m)]",
        ROWCOL: "SICD Row, Column Indices [row (px), col (px)]",
        IROWICOL: "SICD SCP Pixel-Centered Image Indices [irow (px), icol (px)]",
        XROWYCOL: "SICD SCP Centered Image Coordinates [xrow (m), ycol (m)]",
    }
    formats = {
        "WKT": "Well-known text representation of geometry",
        "GeoJSON": f"GeoJSON. Coordinates are assumed to be '{gu.LONLAT}' or '{gu.LONLATHAE}'",
        "raw": "Space or comma separated numbers representing a single point",
    }

    all_cs_options = {
        "auto": "Determine based on format. (Only supported by GeoJSON)"
    } | coordinate_systems

    name_length = max(len(key) for key in all_cs_options.keys())
    epilog = "Supported coordinate systems:\n" + "\n".join(
        f"{key:{name_length}}  {value}" for key, value in all_cs_options.items()
    )

    format_length = max(len(key) for key in formats.keys())
    epilog += "\n\nSupported coordinate formats:\n" + "\n".join(
        f"{key:{format_length}}  {value}" for key, value in formats.items()
    )

    parser = argparse.ArgumentParser(
        description="Convert coordinates to another coordinate system",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "coordinates",
        help="String or filename containing coordinates. '-' reads from stdin",
    )
    parser.add_argument(
        "--from-cs", choices=list(all_cs_options.keys()), default="auto"
    )
    parser.add_argument(
        "--to-cs", choices=list(coordinate_systems.keys()), required=True
    )
    parser.add_argument("--output-format", choices=["WKT"], default="WKT")
    parser.add_argument(
        "--sicd",
        help="SICD NITF or XML filename. Enables rowcol, irowicol, and xrowycol. Assumes SCP height when projecting.",
    )
    config = parser.parse_args(args)

    graph = create_conversion_graph(config.sicd)

    geometry, from_cs = gu.read_coordinates(config.coordinates)
    if config.from_cs != "auto":
        from_cs = config.from_cs

    if from_cs is None:
        raise RuntimeError("Unknown input coordinate system")

    geometry = convert(graph, from_cs, config.to_cs, geometry)

    print(gu.as_wkt(geometry))


if __name__ == "__main__":
    main()
