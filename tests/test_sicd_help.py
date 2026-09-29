import pathlib

import lxml.etree
import numpy as np
import sarkit.sicd as sksicd

import sarkit_processing.sicd_help as skp_sicdhelp

sicd_xml_path = (
    pathlib.Path(__file__).absolute().parents[1] / "data/example-sicd-1.3.0.xml"
)


def test_recompute_deltak():
    # Contrive a test case to hit the various optional metadata paths
    sicdxml = lxml.etree.parse(sicd_xml_path)
    schema = lxml.etree.XMLSchema(
        file=sksicd.VERSION_INFO[lxml.etree.QName(sicdxml.getroot()).namespace][
            "schema"
        ]
    )
    schema.assertValid(sicdxml)
    ew = sksicd.ElementWrapper(sicdxml.getroot())
    # ValidData < Image < FullImage
    ew["ImageData"].from_dict(
        {
            "NumRows": 5,
            "NumCols": 7,
            "FirstRow": 2,
            "FirstCol": 2,
            "FullImage": {"NumRows": 8, "NumCols": 10},
            "SCPPixel": [0, 0],
            "ValidData": [[3, 3], [3, 7], [5, 7], [5, 3]],
        }
    )
    ew["Grid"]["Row"].from_dict(
        {"SS": 1.0, "ImpRespBW": 0.1, "DeltaKCOAPoly": [[0.0, 0.0], [1e-2, 0.0]]}
    )
    ew["Grid"]["Col"].from_dict({"SS": 2.0, "ImpRespBW": 0.2})
    del ew["Grid"]["Col"]["DeltaKCOAPoly"]

    def recompute():
        newxml = skp_sicdhelp.recompute_deltak(sicdxml)
        schema.assertValid(newxml)
        gridew = sksicd.ElementWrapper(newxml.find("{*}Grid"))
        return [(gridew[rc]["DeltaK1"], gridew[rc]["DeltaK2"]) for rc in ("Row", "Col")]

    def within(a, b):
        return (a[0] > b[0]) and (a[1] < b[1])

    rowspan_nyq = np.array([-0.5, 0.5]) / ew["Grid"]["Row"]["SS"]
    rowspan_nyq = np.array([-0.5, 0.5]) / ew["Grid"]["Col"]["SS"]
    rowspan_orig, colspan_orig = recompute()
    assert within(rowspan_orig, rowspan_nyq)
    assert within(colspan_orig, rowspan_nyq)

    # remove ValidData
    del ew["ImageData"]["ValidData"]
    rowspan_novalid, colspan_novalid = recompute()
    assert within(rowspan_orig, rowspan_novalid)
    assert np.array_equal(colspan_novalid, colspan_orig)

    # make Image = FullImage
    ew["ImageData"].from_dict(
        {
            "NumRows": ew["ImageData"]["FullImage"]["NumRows"],
            "NumCols": ew["ImageData"]["FullImage"]["NumCols"],
            "FirstRow": 0,
            "FirstCol": 0,
        }
    )
    rowspan_fullimg, colspan_fullimg = recompute()
    assert within(rowspan_novalid, rowspan_fullimg)
    assert np.array_equal(colspan_fullimg, colspan_orig)

    # alias case
    ew["Grid"]["Col"]["ImpRespBW"] = 1.0
    rowspan_aliascol, colspan_aliascol = recompute()
    assert rowspan_aliascol == rowspan_fullimg
    assert np.array_equal(colspan_aliascol, rowspan_nyq)
