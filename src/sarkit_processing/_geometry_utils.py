import sys

import numpy as np
import shapely

try:
    from smart_open import open
except ImportError:
    pass

LATLON = "latlon"
LATLONHAE = "latlonhae"
LONLAT = "lonlat"
LONLATHAE = "lonlathae"
ECEF = "ecef"


def swap_latlon(ll):
    ll = np.array(ll)
    ll[..., [1, 0]] = ll[..., [0, 1]]
    return ll


def _decode_geojson(encoded):
    geometry = shapely.from_geojson(encoded)
    if geometry.has_z:
        coord_sys = LONLATHAE
    else:
        coord_sys = LONLAT
    return geometry, coord_sys


def _decode_numbers(string):
    if "," in string:
        sep = ","
    else:
        sep = " "
    shape = shapely.Point(np.fromstring(string, np.float64, count=-1, sep=sep))
    return shape, None


def decode_coordinates(coordinates):
    """
    Decode a string of encoded coordinates

    Parameters
    ----------
    coordinates : str

    Returns
    -------
    shapely.Geometry
        Shapely geometry containing the coordinates
    str or None
        Coordinate system of the coordinates.  None if unknown.
    """
    formats = [
        {
            "name": "geojson",
            "decoder": _decode_geojson,
            "exceptions": (shapely.errors.GEOSException,),
        },
        {
            "name": "wkt",
            "decoder": lambda x: (shapely.from_wkt(x), None),
            "exceptions": (shapely.errors.GEOSException,),
        },
        {
            "name": "raw",
            "decoder": _decode_numbers,
            "exceptions": (
                ValueError,
                shapely.errors.GEOSException,
            ),
        },
    ]
    for fmt in formats:
        try:
            geometry, coord_sys = fmt["decoder"](coordinates)
            break
        except fmt["exceptions"]:
            pass
    else:
        raise RuntimeError("Unhandled format")

    return geometry, coord_sys


def read_coordinates(filename_or_string):
    """Read and decode coordinates

    Parameters
    ----------
    filename_or_string : str
        Can be a string containing coordinates, path to file containing coordinates, or "-".
        When "-", coordinates will be read from stdin

    Returns
    -------
    shapely.Geometry
        Shapely geometry containing the coordinates
    str or None
        Coordinate system of the coordinates.  None if unknown.

    """
    contents = filename_or_string

    if filename_or_string == "-":
        contents = sys.stdin.read()
    else:
        try:
            with open(filename_or_string, "r") as file:
                contents = file.read()
        except (OSError, FileNotFoundError):
            pass

    return decode_coordinates(contents)


def as_wkt(geometry):
    return shapely.to_wkt(geometry, rounding_precision=-1)
