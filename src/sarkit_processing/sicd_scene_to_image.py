import argparse
import sys
import textwrap

import numpy as np
import sarkit.sicd as sksicd
import sarkit.wgs84
import shapely
import shapely.geometry as shg

from sarkit_processing import _cli, _io

try:
    from smart_open import open
except ImportError:
    pass


def shapely_scene_to_image(xmltree, geometry):
    """
    Project a shapely geometry into a SICD image

    Parameters
    ----------
    xmltree: lxml.etree.ElementTree
        SICD metadata
    geometry : shapely.Geometry
        geometry shape containing ECEF (WGS84 cartesian) X Y Z coordinates

    Returns
    -------
    shapely.Geometry
        Geometry containing SCP centered coordinates (xrow, ycol).
        Non-collection geometries are converted to empty GeometryCollections if not all coordinates project successfully.
    """

    if hasattr(geometry, "geoms"):  # GeometryCollection and "Multi"
        geoms = [shapely_scene_to_image(xmltree, geo) for geo in geometry.geoms]
        return type(geometry)(geoms)

    def _transform(coordinates):
        image_points, _, success = sksicd.scene_to_image(xmltree, coordinates)
        if not success:
            raise RuntimeError("failed to project all coordinates")
        # Must return same shape as input.  Add a column of 0s.
        return np.append(
            image_points, np.broadcast_to(0, (image_points.shape[0], 1)), -1
        )

    try:
        return shapely.force_2d(shapely.transform(geometry, _transform, include_z=True))
    except RuntimeError:
        pass

    return shg.GeometryCollection()


def _swap_latlon(ll):
    ll = np.array(ll)
    ll[..., [1, 0]] = ll[..., [0, 1]]
    return ll


def _parse_coordinates(coordinates):
    def load_geojson_string(string):
        return shapely.from_geojson(string), "lonlat"

    def load_wkt_string(string):
        return shapely.from_wkt(string), None

    def load_csv_string(string):
        if "," in string:
            sep = ","
        else:
            sep = " "
        shape = shapely.Point(np.fromstring(string, np.float64, count=-1, sep=sep))
        return shape, None

    loaders = [
        (load_geojson_string, (shapely.errors.GEOSException,)),
        (load_wkt_string, (shapely.errors.GEOSException,)),
        (
            load_csv_string,
            (
                ValueError,
                shapely.errors.GEOSException,
            ),
        ),
    ]

    geoshape = None
    from_cs = None
    for loader, allowed_exceptions in loaders:
        try:
            geoshape, from_cs = loader(coordinates)
            break
        except allowed_exceptions:
            pass

    if geoshape is None:
        raise RuntimeError("Failed to interpret coordinates")

    return geoshape, from_cs


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Project points into a SICD",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _cli.allow_floating_point_arguments(parser)
    parser.add_argument("sicd_file", help="SICD filename. May be NITF or XML.")
    parser.add_argument(
        "--from-cs",
        choices=["auto", "lonlat", "ecef"],
        default="auto",
        help=textwrap.dedent("""\
                Input coordinate system:
                'auto': determined by input format (eg. GeoJSON)
                'lonlat': WGS84 geodetic decimal degrees with optional height (defaults to SCP height)
                'ecef': WGS84 cartesian in meters"""),
    )
    parser.add_argument(
        "--to-cs",
        choices=["rowcol", "xrowycol"],
        default="rowcol",
        help="Output coordinate system",
    )
    parser.add_argument(
        "coordinates",
        nargs="+",
        help="Coordinates to convert. May be a filename or string. '-' reads from stdin. Supports GeoJSON, WKT, and space-separated values",
    )
    config = parser.parse_args(args)

    if config.coordinates[0] == "-":
        coordinates = sys.stdin.read()
    else:
        try:
            with open(config.coordinates[0], "r") as file:
                coordinates = file.read()
        except (OSError, FileNotFoundError):
            coordinates = " ".join(config.coordinates)

    geoshape, from_cs = _parse_coordinates(coordinates)

    if config.from_cs == "auto":
        if from_cs is None:
            raise RuntimeError("From coordinate system must be specified")
    else:
        from_cs = config.from_cs

    xmltree = _io.read_sicd_xml(config.sicd_file)
    if from_cs == "lonlat":
        scp_hae = sksicd.XmlHelper(xmltree).load("{*}GeoData/{*}SCP/{*}LLH/{*}HAE")
        geoshape = shapely.force_3d(geoshape, scp_hae)
        geoshape = shapely.transform(
            geoshape,
            lambda x: sarkit.wgs84.geodetic_to_cartesian(_swap_latlon(x)),
            include_z=True,
        )

    xy = shapely_scene_to_image(xmltree, geoshape)

    if xy is None:
        raise RuntimeError("No coordinates successfully projected")

    if config.to_cs == "xrowycol":
        result = xy
    else:
        result = shapely.transform(xy, lambda x: sksicd.xrowycol_to_rowcol(xmltree, x))

    print(shapely.to_wkt(result, rounding_precision=-1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
