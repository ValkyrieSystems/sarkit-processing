import numpy as np
import pytest
import sarkit.cphd as skcphd

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
