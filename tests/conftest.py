import pathlib

import lxml.etree
import numpy as np
import pytest
import sarkit.sicd as sksicd

DATAPATH = pathlib.Path(__file__).parents[1] / "data"

good_sicd_xml_path = DATAPATH / "example-sicd-1.3.0.xml"


def _random_array(shape, dtype, reshape=True):
    rng = np.random.default_rng()
    retval = np.frombuffer(
        rng.bytes(np.prod(shape) * dtype.itemsize), dtype=dtype
    ).copy()

    def _zerofill(arr):
        if arr.dtype.names is None:
            arr[~np.isfinite(arr)] = 0
        else:
            for name in arr.dtype.names:
                _zerofill(arr[name])

    _zerofill(retval)
    return retval.reshape(shape) if reshape else retval


@pytest.fixture(scope="session")
def example_sicd(tmp_path_factory):
    sicd_etree = lxml.etree.parse(good_sicd_xml_path)
    tmp_sicd = (
        tmp_path_factory.mktemp("data") / good_sicd_xml_path.with_suffix(".sicd").name
    )
    sec = {"security": {"clas": "U"}}
    sicd_meta = sksicd.NitfMetadata(
        xmltree=sicd_etree,
        file_header_part={"ostaid": "nowhere"} | sec,
        im_subheader_part={"isorce": "this sensor"} | sec,
        de_subheader_part=sec,
    )
    nrows = int(sicd_etree.findtext("{*}ImageData/{*}NumRows"))
    ncols = int(sicd_etree.findtext("{*}ImageData/{*}NumCols"))
    pixel_type = sicd_etree.findtext("{*}ImageData/{*}PixelType")
    dtype = sksicd.PIXEL_TYPES[pixel_type]["dtype"]
    with open(tmp_sicd, "wb") as f, sksicd.NitfWriter(f, sicd_meta) as w:
        w.write_image(_random_array((nrows, ncols), dtype))
    yield tmp_sicd
