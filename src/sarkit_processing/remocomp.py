"""Utilities for moving CPHD motion compensation"""

import numba
import numpy as np
import numpy.typing as npt
import sarkit.cphd as skcphd
import scipy.constants
from numpy.lib import recfunctions as rfn

from . import atmosphere as atmo


def remocomp_array(
    sig: np.ndarray,
    pvps: np.ndarray,
    sgn: int,
    new_srp_pos: npt.ArrayLike,
    new_td_tropo_srp: npt.ArrayLike,
    *,
    sig_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the motion compensation from one target to another for signal & PVP array pair.

    Note: only FX-domain CPHDs are currently supported.

    Parameters
    ----------
    sig : (nv, ns) ndarray
        Input signal array of dtype=np.complex64 to compensate.
    pvps : (ns,) ndarray
        Input PVP array.
        AmpSF is ignored and should have already been applied.
    sgn : {-1, 1}
        CPHD phase sign parameter
    new_srp_pos : (..., 3) array_like
        New stabilization reference point location with ECEF X, Y, Z components (m) in last dimension.
    new_td_tropo_srp : (...,) array_like
        New two-way time tropospheric delay in seconds for computing propagation times for the new SRP.
    sig_out : ndarray or None, optional
        Where to write compensated signal array. If ``None``, a new array will be created.

    Returns
    -------
    new_sig : np.ndarray
        Re-compensated signal array of dtype=np.complex64
    new_pvps : np.ndarray
        Re-compensated PVP array.
    """
    # TODO: only handles FX CPHDs
    if sig_out is not None:
        if sig_out.shape != tuple(sig.shape):
            raise ValueError(f"array must have shape {sig.shape}, not {sig_out.shape}")
        if sig_out.dtype != np.complex64:
            raise ValueError(
                f"array must have dtype '{np.complex64}', not '{sig_out.dtype}'"
            )
        new_sig = sig_out
    else:
        new_sig = np.empty(sig.shape, dtype=np.complex64)

    # Update PVPs
    srp_rtt, srp_rdot_avg = _compute_rtt_and_rdot_avg(
        pvps["TxPos"],
        pvps["TxVel"],
        pvps["RcvPos"],
        pvps["RcvVel"],
        pvps["SRPPos"],
    )
    new_rtt, new_rdot_avg = _compute_rtt_and_rdot_avg(
        pvps["TxPos"],
        pvps["TxVel"],
        pvps["RcvPos"],
        pvps["RcvVel"],
        new_srp_pos,
    )

    new_rcv_ulos = pvps["RcvPos"] - new_srp_pos
    new_rcv_ulos /= np.linalg.norm(new_rcv_ulos, axis=-1, keepdims=True)
    new_rcv_rdot_stationary = np.vecdot(pvps["RcvVel"], new_rcv_ulos)
    a_toa_vrcv = new_rcv_rdot_stationary / scipy.constants.speed_of_light

    delta_toa = (new_rtt - srp_rtt) + (new_td_tropo_srp - pvps["TDTropoSRP"])
    delta_toa *= 1 + a_toa_vrcv

    delta_rdot_avg = new_rdot_avg - srp_rdot_avg

    new_pvps = pvps.copy()
    new_pvps["SRPPos"] = new_srp_pos
    new_pvps["TDTropoSRP"] = new_td_tropo_srp
    new_pvps["TOA1"] -= delta_toa
    new_pvps["TOA2"] -= delta_toa
    assert new_pvps.dtype.names is not None  # placate mypy
    if {"TOAE1", "TOAE2"}.issubset(new_pvps.dtype.names):
        new_pvps["TOAE1"] -= delta_toa
        new_pvps["TOAE2"] -= delta_toa

    # Create phase polys
    fx0 = pvps["SC0"]  # TODO: only handles FX CPHDs
    fxss = pvps["SCSS"]
    fxc = (pvps["FX1"] + pvps["FX2"]) / 2.0
    delta_fx = fx0 - fxc

    phase_vs_sample_polys = np.zeros((pvps.size, 3))
    phase_vs_sample_polys[:, 0] = fx0 * delta_toa * (
        1 + pvps["aFDOP"]
    ) + delta_rdot_avg * delta_fx * (pvps["aFRR1"] + delta_fx * pvps["aFRR2"])

    phase_vs_sample_polys[:, 1] = fxss * (
        delta_toa * (1 + pvps["aFDOP"])
        + delta_rdot_avg * (pvps["aFRR1"] + 2 * delta_fx * pvps["aFRR2"])
    )

    phase_vs_sample_polys[:, 2] = fxss**2 * pvps["aFRR2"] * delta_rdot_avg

    phase_vs_sample_polys *= -sgn

    # Apply phase polys to signal
    _apply_phase_polys(sig, new_sig, phase_vs_sample_polys)
    return new_sig, new_pvps


def remocomp_cphd_chan(
    cphd_reader: skcphd.Reader,
    ch_id: str,
    new_srp_pos: npt.ArrayLike,
    tropo_n0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the motion compensation from one target to another for a given CPHD channel.

    Note: only FX-domain CPHDs are currently supported.

    Parameters
    ----------
    cphd_reader : sarkit.cphd.Reader
        Open CPHD reader object
    ch_id : str
       Identifier of channel to remocomp
    new_srp_pos : (..., 3) array_like
        New stabilization reference point location with ECEF X, Y, Z components (m) in last dimension
    tropo_n0 : float or None, optional
        Refractivity (N) of the troposphere at HAE=0 to use to compute the delay to the new SRP due to the troposphere.
        If ``None``, the value from the CPHD XML is used, when present. Otherwise, a built-in constant is used.

    Returns
    -------
    sig : np.ndarray
        Re-compensated signal array of dtype=np.complex64
    pvps : np.ndarray
        Re-compensated PVP array.
        If the input PVPs contain an AmpSF parameter, it is consumed and not preserved.
    """
    if cphd_reader.metadata.xmltree.findtext("{*}Global/{*}DomainType") != "FX":
        raise NotImplementedError("Only FX CPHDs are currently supported.")

    sig, pvps = cphd_reader.read_channel(ch_id)
    if sig.dtype.names is None:
        assert sig.dtype.newbyteorder("=") == np.dtype("c8")
        sig = sig.astype(np.complex64, copy=False)
    else:
        sig = sig["real"].astype(np.float32) + 1j * sig["imag"].astype(np.float32)
    assert pvps.dtype.names is not None  # placate mypy
    if "AmpSF" in pvps.dtype.names:
        sig *= np.abs(pvps["AmpSF"][:, np.newaxis])
        pvps = rfn.drop_fields(pvps, "AmpSF", usemask=False)

    ew = skcphd.ElementWrapper(cphd_reader.metadata.xmltree.getroot())

    if tropo_n0 is None:
        tropo_n0 = ew["Global"]["TropoParameters"].get("N0", atmo.DEFAULT_N0)
        if ew["Global"]["TropoParameters"].get("RefHeight", "ZERO") != "ZERO":
            tropo_n0 = atmo.ellipsoid_refractivity(
                tropo_n0, ew["SceneCoordinates"]["IARP"]["LLH"][2]
            )
    new_td_tropo_srp = atmo.one_way_tropo_delay(
        new_srp_pos, pvps["TxPos"], tropo_n0
    ) + atmo.one_way_tropo_delay(new_srp_pos, pvps["RcvPos"], tropo_n0)

    return remocomp_array(
        sig,
        pvps,
        ew["Global"]["SGN"],
        new_srp_pos,
        new_td_tropo_srp,
        sig_out=sig,
    )


def _compute_rtt_and_rdot_avg(txpos, txvel, rcvpos, rcvvel, pt):
    tx_los = txpos - pt
    tx_range = np.linalg.norm(tx_los, axis=-1)
    rcv_los = rcvpos - pt
    rcv_range = np.linalg.norm(rcv_los, axis=-1)
    rtt = (tx_range + rcv_range) / scipy.constants.speed_of_light

    tx_rdot = np.vecdot(txvel, tx_los / tx_range[..., np.newaxis])
    rcv_rdot = np.vecdot(rcvvel, rcv_los / rcv_range[..., np.newaxis])
    rdot_avg = (tx_rdot + rcv_rdot) / 2.0
    return rtt, rdot_avg


@numba.njit
def _phasor(phase_cyc):
    phase_rad = phase_cyc * 2 * np.pi
    return np.complex64(np.cos(phase_rad) + 1j * np.sin(phase_rad))


@numba.njit
def _quad_phase(signal_in, signal_out, phase_poly):
    c0, c1, c2 = phase_poly
    for s in range(signal_in.shape[-1]):
        signal_out[s] = signal_in[s] * _phasor(c0 + s * (c1 + c2 * s))


@numba.njit(parallel=True)
def _apply_phase_polys(signal_in, signal_out, phase_polys):
    for n in numba.prange(signal_in.shape[0]):
        _quad_phase(signal_in[n], signal_out[n], phase_polys[n])
