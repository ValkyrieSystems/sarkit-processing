import numpy as np
import sarkit.sicd as sksicd
import shapely
import shapely.geometry as shg


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
