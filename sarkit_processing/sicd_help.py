"""Module for miscellaneous SICD functions."""

import copy
import functools

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import sarkit.sicd as sksicd
import shapely


def _get_samples_in_poly(poly: shapely.Polygon, grid_size: int = 11) -> np.ndarray:
    """Return samples that intersect a polygon."""
    bounds = np.asarray(poly.bounds).reshape(2, 2)  # [[xmin, ymin], [xmax, ymax]]
    mesh = np.stack(
        np.meshgrid(
            np.linspace(bounds[0, 0], bounds[1, 0], grid_size),
            np.linspace(bounds[0, 1], bounds[1, 1], grid_size),
        ),
        axis=-1,
    )
    inner_mesh = shapely.get_coordinates(poly.intersection(shapely.multipoints(mesh)))
    poly_vertices = shapely.get_coordinates(poly.exterior)[:-1]
    return np.concatenate(
        [inner_mesh, poly_vertices],
        axis=0,
    )


def recompute_deltak(sicd_xmltree: lxml.etree.ElementTree) -> lxml.etree.ElementTree:
    """Return a copy of a SICD XML with recomputed Grid//DeltaK1 and Grid//DeltaK2."""
    sicd_xmltree = copy.deepcopy(sicd_xmltree)
    ew = sksicd.ElementWrapper(sicd_xmltree.getroot())

    if sicd_xmltree.find("{*}Grid/*/{*}DeltaKCOAPoly") is not None:
        nrows = ew["ImageData"]["NumRows"]
        ncols = ew["ImageData"]["NumCols"]
        r0 = ew["ImageData"]["FirstRow"]
        c0 = ew["ImageData"]["FirstCol"]
        nrows_fi = ew["ImageData"]["FullImage"]["NumRows"]
        ncols_fi = ew["ImageData"]["FullImage"]["NumCols"]

        polygons_rowcol = [
            shapely.box(-0.5, -0.5, nrows_fi - 0.5, ncols_fi - 0.5),
            shapely.box(r0 - 0.5, c0 - 0.5, r0 + nrows - 0.5, c0 + ncols - 0.5),
        ]
        validdata = ew["ImageData"].get("ValidData", None)
        if validdata is not None:
            polygons_rowcol.append(shapely.Polygon(validdata).buffer(0.5))
        polygons_xrowycol = [
            shapely.transform(
                p, functools.partial(sksicd.rowcol_to_xrowycol, sicd_xmltree)
            )
            for p in polygons_rowcol
        ]
        polygon_to_sample = shapely.intersection_all(polygons_xrowycol)
        pts_xrowycol = _get_samples_in_poly(polygon_to_sample)

    for rowcol in ("Row", "Col"):
        griddir = ew["Grid"][rowcol]
        dkcoapoly = griddir.get("DeltaKCOAPoly", None)
        if dkcoapoly is not None:
            dkcoa = npp.polyval2d(pts_xrowycol[:, 0], pts_xrowycol[:, 1], dkcoapoly)
            dkcoa_min = dkcoa.min()
            dkcoa_max = dkcoa.max()
        else:
            dkcoa_min = 0.0
            dkcoa_max = 0.0
        dk1 = dkcoa_min - (griddir["ImpRespBW"] / 2.0)
        dk2 = dkcoa_max + (griddir["ImpRespBW"] / 2.0)

        # saturate aliased spectrum
        dk_nyq = 0.5 / griddir["SS"]
        if (dk1 < -dk_nyq) or (dk2 > dk_nyq):
            dk1 = -dk_nyq
            dk2 = dk_nyq

        griddir["DeltaK1"] = dk1
        griddir["DeltaK2"] = dk2

    return sicd_xmltree
