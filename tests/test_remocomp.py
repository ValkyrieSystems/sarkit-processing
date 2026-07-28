import lxml.etree
import numpy as np
import pytest
import sarkit.cphd as skcphd
from numpy.lib import recfunctions as rfn

import sarkit_processing.remocomp as remo


def fake_data():
    pvptype = skcphd.get_defined_pvp_dtype(
        "http://api.nsgreg.nga.mil/schema/cphd/1.1.0"
    )
    pvps = np.zeros(1, pvptype)

    srp = np.array([6378137.0, 0, 0])
    graze_angle = 0.5
    slant_range = 1e6
    pvps["TxPos"] = (
        srp
        + np.cos(graze_angle) * slant_range * np.array([0, 0, 1])
        + np.sin(graze_angle) * slant_range * np.array([1, 0, 0])
    )
    pvps["TxVel"] = [0.0, 7.5e3, 0.0]
    pvps["RcvPos"] = (
        srp
        + np.cos(graze_angle) * slant_range * np.array([0, 1, 0])
        + np.sin(graze_angle) * slant_range * np.array([1, 0, 0])
    )
    pvps["RcvVel"] = [0.0, np.sin(0.1) * 7.5e3, np.cos(0.1) * 7.5e3]
    pvps["FX1"] = 9e9
    pvps["FX2"] = 9e9
    pvps["SCSS"] = 1e9 / 1023
    pvps["SC0"] = 9e9
    pvps["SRPPos"] = srp
    pvps["TOA1"] = -1e-4
    pvps["TOA2"] = 1e-4
    pvps["TOAE1"] = -1.1e-4
    pvps["TOAE2"] = 1.1e-4
    pvps["aFDOP"] = 8.5e-6
    pvps["aFRR1"] = 2.6e-12
    pvps["aFRR2"] = 2.6e-22

    sig = np.ones((1, 24), dtype=np.complex64)
    return sig, pvps


def fake_cphd(cphd_path, *, sgn=-1, sig_format="CF8", tropo=None):
    sig, pvps = fake_data()
    ew = skcphd.ElementWrapper(
        lxml.etree.Element("{http://api.nsgreg.nga.mil/schema/cphd/1.1.0}CPHD")
    )
    ew["CollectionID"]["Classification"] = "UNCLASSIFIED"
    ew["CollectionID"]["ReleaseInfo"] = "UNRESTRICTED"
    ew["Global"]["DomainType"] = "FX"
    ew["Global"]["SGN"] = sgn
    if tropo is not None:
        ew["Global"]["TropoParameters"] = tropo
    ew["SceneCoordinates"]["IARP"]["LLH"] = (
        0.0,
        0.0,
        824,  # important thing is that HAE > 0
    )
    ew["Data"]["SignalArrayFormat"] = sig_format
    if sig_format == "CF8":
        pvps = rfn.drop_fields(pvps, "AmpSF")
    else:
        sig = np.zeros(
            sig.shape, dtype=skcphd.binary_format_string_to_dtype(sig_format)
        )
        sig["real"] = 2
        pvps["AmpSF"] = 0.5
    ew["Data"]["NumBytesPVP"] = pvps.dtype.itemsize
    datachan = ew["Data"].add("Channel")
    datachan["Identifier"] = "fake"
    datachan["NumVectors"] = sig.shape[0]
    datachan["NumSamples"] = sig.shape[1]
    datachan["SignalArrayByteOffset"] = 0
    datachan["PVPArrayByteOffset"] = 0

    ew["Channel"]["RefChId"] = datachan["Identifier"]
    ew["PVP"] = skcphd.dtype_to_pvp_element(
        "http://api.nsgreg.nga.mil/schema/cphd/1.1.0", pvps.dtype
    )

    meta = skcphd.Metadata(xmltree=ew.elem.getroottree())
    with open(cphd_path, "wb") as f, skcphd.Writer(f, meta) as w:
        w.write_pvp("fake", pvps)
        w.write_signal("fake", sig)


def test_remocomp_array():
    sig, pvps = fake_data()

    # Running with old SRP should effectively be a no-op
    sig_noop, pvps_noop = remo.remocomp_array(sig, pvps, -1, pvps["SRPPos"], 0.0)
    assert np.allclose(sig, sig_noop)
    for name in pvps.dtype.names:
        assert np.allclose(pvps[name], pvps_noop[name])

    # Run with new SRP, SGN=-1
    new_srp = pvps["SRPPos"] + [100.0, 0.0, 0.0]
    sig_neg, pvps_neg = remo.remocomp_array(sig, pvps, -1, new_srp, 0.0)
    assert np.allclose(pvps_neg["SRPPos"], new_srp)
    for field in ("TOA1", "TOA2", "TOAE1", "TOAE2"):
        assert not np.allclose(pvps_neg[field], pvps[field])
    assert not np.allclose(sig, sig_neg)

    # Run with new SRP, SGN=+1
    sig_pos, pvps_pos = remo.remocomp_array(sig, pvps, +1, new_srp, 0.0)
    for name in pvps.dtype.names:
        assert np.array_equal(pvps_neg[name], pvps_pos[name])
    assert np.allclose(sig, sig_neg * sig_pos)

    # Run with tropo
    sig_tropo, pvps_tropo = remo.remocomp_array(sig, pvps, -1, pvps["SRPPos"], 315.0)
    sig_moretropo, pvps_moretropo = remo.remocomp_array(
        sig, pvps, -1, pvps["SRPPos"], 320.0
    )
    assert pvps_moretropo["TDTropoSRP"] > pvps_tropo["TDTropoSRP"]
    assert not np.allclose(sig, sig_tropo)
    assert not np.allclose(sig, sig_moretropo)

    # Test sig_out kwarg
    sig_out = np.empty_like(sig)
    sig_noop2, pvps_noop2 = remo.remocomp_array(
        sig, pvps, -1, pvps["SRPPos"], 0.0, sig_out=sig_out
    )
    for name in pvps_noop.dtype.names:
        assert np.array_equal(pvps_noop[name], pvps_noop2[name])
    assert sig_noop2 is sig_out
    assert np.array_equal(sig_noop, sig_noop2)

    with pytest.raises(ValueError, match="must have shape"):
        remo.remocomp_array(sig, pvps, -1, pvps["SRPPos"], 0.0, sig_out=sig_out[:, :1])
    with pytest.raises(ValueError, match="must have dtype"):
        remo.remocomp_array(
            sig, pvps, -1, pvps["SRPPos"], 0.0, sig_out=sig_out.view(np.int64)
        )


def test_remocomp_cphd_sgn(tmp_path):
    neg_cphd = tmp_path / "neg.cphd"
    fake_cphd(neg_cphd, sgn=-1)
    with open(neg_cphd, "rb") as f, skcphd.Reader(f) as r:
        neg_sig, neg_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    pos_cphd = tmp_path / "pos.cphd"
    fake_cphd(pos_cphd, sgn=+1)
    with open(pos_cphd, "rb") as f, skcphd.Reader(f) as r:
        pos_sig, pos_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    for name in neg_pvps.dtype.names:
        assert np.array_equal(neg_pvps[name], pos_pvps[name])
    assert not np.allclose(neg_sig, pos_sig)


def test_remocomp_cphd_sigformat(tmp_path):
    cf8_cphd = tmp_path / "cf8.cphd"
    fake_cphd(cf8_cphd, sig_format="CF8")
    with open(cf8_cphd, "rb") as f, skcphd.Reader(f) as r:
        cf8_sig, cf8_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    ci4_cphd = tmp_path / "ci4.cphd"
    fake_cphd(ci4_cphd, sig_format="CI4")
    with open(ci4_cphd, "rb") as f, skcphd.Reader(f) as r:
        ci4_sig, ci4_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    for name in cf8_pvps.dtype.names:
        if name == "AmpSF":
            assert name not in ci4_pvps.dtype.name
        else:
            assert np.array_equal(ci4_pvps[name], ci4_pvps[name])
    assert np.array_equal(cf8_sig, ci4_sig)


def test_remocomp_cphd_no_tropo(tmp_path):
    no_tropo_cphd = tmp_path / "notropo.cphd"
    fake_cphd(no_tropo_cphd, tropo=None)
    with open(no_tropo_cphd, "rb") as f, skcphd.Reader(f) as r:
        orig_pvps = r.read_pvps("fake")
        _, no_tropo_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    assert orig_pvps["TDTropoSRP"] == 0.0
    assert no_tropo_pvps["TDTropoSRP"] > 0.0


def test_remocomp_cphd_tropo_refheight(tmp_path):
    zero_cphd = tmp_path / "zero.cphd"
    fake_cphd(zero_cphd, tropo={"N0": 320.0, "RefHeight": "ZERO"})
    with open(zero_cphd, "rb") as f, skcphd.Reader(f) as r:
        _, zero_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    iarp_cphd = tmp_path / "iarp.cphd"
    fake_cphd(iarp_cphd, tropo={"N0": 320.0, "RefHeight": "IARP"})
    with open(iarp_cphd, "rb") as f, skcphd.Reader(f) as r:
        _, iarp_pvps = remo.remocomp_cphd_chan(r, "fake", [6378137.0, 100.0, 0])

    assert float(r.metadata.xmltree.find(".//{*}IARP/{*}LLH")[-1].text) > 0.0
    assert iarp_pvps["TDTropoSRP"] > zero_pvps["TDTropoSRP"]
