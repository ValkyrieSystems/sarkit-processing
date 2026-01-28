"""
Deskew (apply phase polynomial) to SICDs.
"""

import copy

import lxml.etree
import numba
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt
import sarkit.sicd as sksicd


@numba.njit(parallel=True)
def _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss):
    """numba parallelized phase poly application"""
    out = np.empty(array.shape, array.dtype)
    for rowidx in numba.prange(out.shape[0]):
        row_val = row_0 + rowidx * row_ss
        col_poly = phase_poly[-1, :]
        for ndx in range(phase_poly.shape[0] - 1, 0, -1):
            col_poly = col_poly * row_val + phase_poly[ndx - 1, :]
        for colidx in range(out.shape[1]):
            col_val = col_0 + colidx * col_ss
            phase_val = col_poly[-1]
            for ndx in range(col_poly.shape[0] - 1, 0, -1):
                phase_val = phase_val * col_val + col_poly[ndx - 1]

            out[rowidx, colidx] = array[rowidx, colidx] * np.exp(
                1j * 2 * np.pi * phase_val
            )

    return out


def _update_grid_metadata(phase_poly, sicd_xmltree):
    """Update the metadata following a deskew operation"""
    sicdew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    for dim in ["Row", "Col"]:
        axis_index = {"Row": 0, "Col": 1}[dim]
        axis_mdata = sicdew["Grid"][dim]
        delta_k_coa_poly = axis_mdata.get("DeltaKCOAPoly", np.array([[0.0]]))
        phase_poly_der = npp.polyder(-phase_poly, axis=axis_index) * axis_mdata["Sgn"]

        max_dims = np.amax([delta_k_coa_poly.shape, phase_poly_der.shape], axis=0)
        pad = max_dims - phase_poly_der.shape
        phase_poly_der = np.pad(
            phase_poly_der, ((0, pad[0]), (0, pad[1])), mode="constant"
        )
        pad = max_dims - delta_k_coa_poly.shape
        delta_k_coa_poly = np.pad(
            delta_k_coa_poly, ((0, pad[0]), (0, pad[1])), mode="constant"
        )

        updated_poly = delta_k_coa_poly + phase_poly_der
        axis_mdata["DeltaKCOAPoly"] = updated_poly


def get_deskew_phase_poly(
    sicd_xmltree: lxml.etree.ElementTree, axis: str
) -> npt.NDArray:
    """Return phase polynomial to deskew specified axis

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    phase_poly : ndarray
        Array of phase polynomial coefficients
    """
    axis_index = {"Row": 0, "Col": 1}[axis]
    sicdew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    delta_k_coa_poly = sicdew["Grid"][axis].get("DeltaKCOAPoly", np.array([[0.0]]))

    phase_poly = np.array([[0.0]])
    if np.all(delta_k_coa_poly == 0):
        return phase_poly

    phase_poly = (
        npp.polyint(delta_k_coa_poly, axis=axis_index) * sicdew["Grid"][axis]["Sgn"]
    )
    return phase_poly


def apply_phase_poly(
    array: npt.NDArray, phase_poly: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Metadata aware phase poly application

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    phase_poly : ndarray
        Array of phase polynomial coefficients
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree

    Returns
    -------
    array_out : ndarray
        2D array of adjusted complex pixels
    sicd_xmltree_out : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """
    sicd_xmltree_out = copy.deepcopy(sicd_xmltree)
    sicdew = sksicd.ElementWrapper(sicd_xmltree_out.getroot())
    row_scp, col_scp = sicdew["ImageData"]["SCPPixel"]
    row_ss = sicdew["Grid"]["Row"]["SS"]
    row_0 = (sicdew["ImageData"]["FirstRow"] - row_scp) * row_ss
    col_ss = sicdew["Grid"]["Col"]["SS"]
    col_0 = (sicdew["ImageData"]["FirstCol"] - col_scp) * col_ss

    array_out = _apply_phase_poly(array, phase_poly, row_0, row_ss, col_0, col_ss)
    _update_grid_metadata(phase_poly, sicd_xmltree_out)

    return array_out, sicd_xmltree_out


def deskew(
    array: npt.NDArray, sicd_xmltree: lxml.etree.ElementTree, axis: str
) -> tuple[npt.NDArray, lxml.etree.ElementTree]:
    """Deskew complex data array

    Parameters
    ----------
    array : ndarray
        2D array of complex pixels
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    axis : {'Row', 'Col'}
        Which axis to deskew

    Returns
    -------
    array_deskew : ndarray
        2D array of deskewed complex pixels
    sicd_xmltree_deskew : lxml.etree.ElementTree
        Updated SICD XML ElementTree
    """
    phase_poly = get_deskew_phase_poly(sicd_xmltree, axis)
    if np.all(phase_poly == 0):
        return array, sicd_xmltree
    return apply_phase_poly(array, phase_poly, sicd_xmltree)
