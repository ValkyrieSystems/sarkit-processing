import json

import numpy as np
import pytest
import shapely

import sarkit_processing._geometry_utils as gu


def test_swap_latlon():
    shape = (5, 4, 3, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    swapped = gu.swap_latlon(data)
    assert swapped.shape == data.shape
    assert np.all(swapped[..., 0] == data[..., 1])
    assert np.all(swapped[..., 1] == data[..., 0])
    assert np.all(swapped[..., 2] == data[..., 2])


def test_decode_coordinates():
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
    geometry, coord_sys = gu.decode_coordinates(json.dumps(geojson))
    assert shapely.get_num_geometries(geometry) == 3
    assert shapely.get_num_geometries(geometry.geoms[-1]) == 2
    assert coord_sys == gu.LONLATHAE

    geo3d = shapely.force_3d(
        geometry, 1.1
    )  # shapely has trouble parsing WKT with mixed 2D & 3D
    geometry, coord_sys = gu.decode_coordinates(geo3d.wkt)
    assert shapely.get_num_geometries(geometry) == 3
    assert shapely.get_num_geometries(geometry.geoms[-1]) == 2
    assert coord_sys is None

    geometry, coord_sys = gu.decode_coordinates("1.1, 2.2")
    assert geometry.coords[:] == [(1.1, 2.2)]
    assert coord_sys is None

    geometry, coord_sys = gu.decode_coordinates("1.1, 2.2, 3.3")
    assert geometry.coords[:] == [(1.1, 2.2, 3.3)]
    assert coord_sys is None

    geometry, coord_sys = gu.decode_coordinates("1.1     2.2 3.3")
    assert geometry.coords[:] == [(1.1, 2.2, 3.3)]
    assert coord_sys is None

    with pytest.raises(RuntimeError, match="Unhandled format"):
        gu.decode_coordinates("foobar")
