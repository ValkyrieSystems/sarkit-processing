import math
from typing import Sequence

import lxml.etree
import numba
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt
import sarkit.sicd as sksicd
import scipy.fft as spfft
import scipy.ndimage

from sarkit_processing import _cli, sicd_pixel_type
from sarkit_processing.sicd_deskew import apply_phase_poly, get_deskew_phase_poly

try:
    from smart_open import open
except ImportError:
    pass


def _shift_poly_axis(coeffs: npt.ArrayLike, shift: float, axis: int) -> npt.NDArray:
    """
    Return new coefficients so that:

        OrigPoly(..., x_axis + shift, ...) == NewPoly(..., x_axis, ...)

    coeffs[k0, k1, ...] is the coefficient for:
        x0**k0 * x1**k1 * ...

    Parameters
    ----------
    coeffs : array-like
        N-dimensional polynomial coefficient tensor.
    shift : float
        Shift applied to the selected input axis.
    axis : int
        Axis whose variable is shifted.

    Returns
    -------
    ndarray
        Shifted coefficient tensor.
    """
    coeffs = np.asarray(coeffs)
    out = np.zeros_like(coeffs, dtype=np.result_type(coeffs, shift, float))

    degree = coeffs.shape[axis] - 1

    # Move target polynomial axis to front: shape (degree+1, ...)
    c = np.moveaxis(coeffs, axis, 0)
    o = np.moveaxis(out, axis, 0)

    powers = shift ** np.arange(degree + 1)

    for old_power in range(degree + 1):
        # (x + shift)^old_power = sum new_power terms
        for new_power in range(old_power + 1):
            o[new_power] += (
                c[old_power]
                * math.comb(old_power, new_power)
                * powers[old_power - new_power]
            )

    return out


def _polyscale2d(coeffs: npt.NDArray, scale_x: float, scale_y: float) -> npt.NDArray:
    """
    Returns new polynomial with scaled coordinate axes so that:

        NewPoly(x, y) == OrigPoly(scale_x * x, scale_y * y).

    Parameters
    ----------
    coeffs : ndarray
        2-D polynomial coefficients, where coeffs[i, j] is the coefficient
        of x**i * y**j.
    scale_x : float
        Scale factor applied to the x coordinate.
    scale_y : float
        Scale factor applied to the y coordinate.

    Returns
    -------
    ndarray
        Scaled coefficient tensor.
    """
    ix = np.arange(coeffs.shape[0])[:, np.newaxis]
    iy = np.arange(coeffs.shape[1])[np.newaxis, :]
    scale = (scale_x**ix) * (scale_y**iy)
    return coeffs * scale


@numba.njit(parallel=True)
def _apply_row_phase_poly(array, phase_poly, col_0, col_ss):
    """Apply a separate 1D phase polynomial to each row."""
    out = np.empty_like(array)
    for rowidx in numba.prange(array.shape[0]):
        coeffs = phase_poly[rowidx]
        for colidx in range(array.shape[1]):
            col_val = col_0 + colidx * col_ss
            phase_val = coeffs[-1]
            for ndx in range(coeffs.shape[0] - 1, 0, -1):
                phase_val = phase_val * col_val + coeffs[ndx - 1]
            out[rowidx, colidx] = array[rowidx, colidx] * np.exp(
                1j * 2 * np.pi * phase_val
            )

    return out


def _power(arr):
    return np.abs(arr) ** 2


def _powersum(arr):
    return np.sum(np.abs(arr) ** 2)


def _normalize(arr):
    out = np.zeros_like(arr)
    np.divide(arr, np.abs(arr), out=out, where=arr != 0)
    return out


def _unit(arr):
    return arr / np.linalg.norm(arr, axis=-1)


def _apply_spectral_phase(
    arr, phase_polys, fft_sizes, output_shape, col_0=0, col_ss=1.0
):
    arr = spfft.fftshift(spfft.fft(arr, axis=1, n=fft_sizes[1]), axes=1)
    arr = spfft.fftshift(spfft.fft(arr.T, axis=1, n=fft_sizes[0]), axes=1)
    arr = _apply_row_phase_poly(arr, phase_polys, col_0, col_ss)
    arr = spfft.ifft(spfft.ifftshift(arr, axes=1), axis=1, n=fft_sizes[0])[
        :, : output_shape[0]
    ]
    arr = spfft.ifft(spfft.ifftshift(arr.T, axes=1), axis=1, n=fft_sizes[1])[
        :, : output_shape[1]
    ]

    return arr


def _focus_alias(arr, sicd_xmltree, zone, prf_override):
    sicdew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    if prf_override is not None:
        ipp_poly = [0, prf_override]
    else:
        ipp_poly = sicdew["Timeline"]["IPP"]["Set"][0]["IPPPoly"]  # TODO which poly?

    coa_nom = npp.polyval2d(0, 0, sicdew["Grid"]["TimeCOAPoly"])
    prf_nom = npp.polyval(coa_nom, npp.polyder(ipp_poly))
    krg_ctr_nom = sicdew["Grid"]["Row"]["KCtr"]
    alias_freq_nom = zone * prf_nom
    range_rate_nom = alias_freq_nom / krg_ctr_nom
    arp_poly = sicdew["Position"]["ARPPoly"]
    pos_nom = npp.polyval(coa_nom, arp_poly)
    vel_nom = npp.polyval(coa_nom, npp.polyder(arp_poly))
    scp = sicdew["GeoData"]["SCP"]["ECF"]
    tan_pa_rate = np.linalg.norm(
        np.cross(vel_nom, _unit(scp - pos_nom))
    ) / np.linalg.norm(scp - pos_nom)
    phase_slope_poly = np.array(
        [-alias_freq_nom / tan_pa_rate, range_rate_nom / tan_pa_rate]
    )
    fft_size_0 = spfft.next_fast_len(arr.shape[0])
    fft_size_1 = spfft.next_fast_len(arr.shape[1])
    kaz_ss = 1.0 / (sicdew["Grid"]["Col"]["SS"] * fft_size_1)
    kaz = (np.arange(fft_size_1) - fft_size_1 // 2) * kaz_ss
    krg_ss = 1.0 / (sicdew["Grid"]["Row"]["SS"] * fft_size_0)
    phase_vs_krg_poly = phase_slope_poly * kaz[:, np.newaxis]
    phase_vs_samp_poly = _polyscale2d(
        _shift_poly_axis(phase_vs_krg_poly, krg_ctr_nom, 1), 1.0, krg_ss / krg_ctr_nom
    )
    fft_sizes = (fft_size_0, fft_size_1)
    focus = _apply_spectral_phase(arr, phase_vs_samp_poly, fft_sizes, fft_sizes)

    return focus, fft_sizes, phase_vs_samp_poly


def _remove_alias(arr, sicd_xmltree, zone, thresh, dilation, prf_override):
    # TODO remove intermediate variables
    focus_alias, fft_sizes, phase_polys = _focus_alias(
        arr, sicd_xmltree, zone, prf_override
    )
    normalize = _normalize(arr)
    focus_alias_norm = _apply_spectral_phase(
        normalize, phase_polys, fft_sizes, fft_sizes
    )
    power = _power(focus_alias_norm)

    # Gives array of bool
    over_thresh = power > thresh  # TODO numba
    if dilation > 0:  # TODO verify
        dilated_mask = scipy.ndimage.binary_dilation(
            over_thresh,
            structure=np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=bool),
        )
    else:
        dilated_mask = over_thresh

    removed_power = np.sum(np.abs(focus_alias) ** 2, where=dilated_mask)
    under_thresh_vals = np.where(~dilated_mask, focus_alias, 0)
    focus_img = _apply_spectral_phase(
        under_thresh_vals, -phase_polys, fft_sizes, arr.shape
    )
    kept_data_frac = np.mean(~dilated_mask)

    return focus_img, removed_power, kept_data_frac


def _mitigate(arr, sicd_xmltree, zones, thresh, dilation, prf_override=None):
    removed_power = np.float32(0)
    kept_data_frac = np.float64(1)
    for zone in zones:
        arr, zone_removed_power, zone_kept_data_frac = _remove_alias(
            arr, sicd_xmltree, zone, thresh, dilation, prf_override
        )
        removed_power += zone_removed_power
        kept_data_frac *= zone_kept_data_frac

    return arr, removed_power, kept_data_frac


def _minimumpower(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    return np.where(np.abs(left) <= np.abs(right), left, right)


def _trim_spectrum(arr, sicd_xmltree, axis):
    sicdew = sksicd.ElementWrapper(sicd_xmltree.getroot())
    axis_idx = {"Row": 0, "Col": 1}[axis]
    fft_size = spfft.next_fast_len(arr.shape[axis_idx])
    num_keep = int(
        np.ceil(
            sicdew["Grid"][axis]["ImpRespBW"] * sicdew["Grid"][axis]["SS"] * fft_size
        )
    )
    phase_poly = get_deskew_phase_poly(sicd_xmltree, axis=axis)
    arr, sicd_xmltree = apply_phase_poly(arr, phase_poly, sicd_xmltree)
    if axis == "Row":
        arr = arr.T
    fftf0 = spfft.fft(arr, axis=1, n=fft_size)
    fftf0[:, num_keep // 2 : fft_size - num_keep // 2] = 0
    ffti0 = spfft.ifft(fftf0, axis=1, n=fft_size)[:, : arr.shape[-1]]
    if axis == "Row":
        ffti0 = ffti0.T
    arr, sicd_xmltree = apply_phase_poly(ffti0, -phase_poly, sicd_xmltree)
    return arr, sicd_xmltree


def _trim_spectrum_2d(arr, sicd_xmltree):
    arr, sicd_xmltree = _trim_spectrum(arr, sicd_xmltree, "Col")
    arr, sicd_xmltree = _trim_spectrum(arr, sicd_xmltree, "Row")

    return arr


def prf_alias_removal(
    image: npt.NDArray,
    sicd_xmltree: lxml.etree.ElementTree,
    num_iters: int,
    zones: Sequence[float],
    *,
    threshold: float = 9.0,
    dilate: int = 1,
    prf: float | None = None,
) -> tuple[npt.NDArray, float, float]:
    """
    Remove PRF alias energy

    Parameters
    ----------
    image : ndarray
        The complex array to apply the algorithm.
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML ElementTree
    num_iters : int
        Number of iterations of cleaning.
    zones : sequence of float
        Sequence of alias zones that should be cleaned.  The zone values need not be integers.
    threshold : float, optional
        Threshold to use when creating NIFT mask (units of stddev) (Default: 9.0).
    dilate : int, optional
        Number of pixels to dilate the masks (Default: 1).
    prf : float or None, optional
        Pulse repetition frequency (Hz).  (Default: None).
        If ``None``, the SICD's Timeline/IPP parameters are used.

    Returns
    -------
    out : ndarray
        The array with alias energy removed.
    removed_pwr_frac : float
        The fraction of the power that was removed.
    removed_data_frac: float
        The fraction of the bins that were removed.
    """
    out = before = image.copy()
    ps = _powersum(image)
    removed_power = np.float32(0)
    kept_data_frac = np.float64(1)
    for _ in range(num_iters):
        out, iter_removed_power, iter_kept_data_frac = _mitigate(
            out, sicd_xmltree, zones, threshold, dilate, prf
        )
        removed_power += iter_removed_power
        kept_data_frac *= iter_kept_data_frac
        # TODO: Verify trim
        out = _minimumpower(out, before)
        out = _trim_spectrum_2d(out, sicd_xmltree)

    removed_pwr_frac = removed_power / ps
    removed_data_frac = 1.0 - kept_data_frac

    return out, removed_pwr_frac, removed_data_frac


class _SicdAliasSubcommand(_cli.Subcommand):
    def get_argument_parser_kwargs(self):
        return dict(
            description="Clean up PRF alias energy from a SICD",
        )

    def add_arguments(self, parser):
        parser.add_argument("input_sicd_filename")
        parser.add_argument("output_sicd_filename")
        parser.add_argument(
            "--threshold",
            type=float,
            default=9.0,
            help=(
                "Threshold to use when creating NIFT mask (units of stddev) "
                "(default: %(default)g)"
            ),
        )
        parser.add_argument(
            "--num-iters",
            type=int,
            default=2,
            help="Number of iterations of cleaning to do (default: %(default)d)",
        )
        parser.add_argument(
            "--zones",
            type=float,
            default=[-2, -1, 1, 2],
            nargs="+",
            help="PRF multipliers to search for alias energy (default: %(default)s)",
        )
        parser.add_argument(
            "--symmetric",
            action="store_true",
            help="Search positive and negative of zones (default: listed zones only, not their inverse)",
        )
        parser.add_argument(
            "--dilate",
            type=int,
            default=1,
            help="Number of pixels to dilate the masks (default: %(default)d)",
        )
        parser.add_argument(
            "--prf",
            type=float,
            default=None,
            help="Override the PRF with a fixed value",
        )

    def run_command(self, config):
        zones = config.zones.copy()
        if config.symmetric:
            zones += [-zone for zone in config.zones]
        zones = sorted(set(zones))

        with (
            open(config.input_sicd_filename, "rb") as file,
            sksicd.NitfReader(file) as reader,
        ):
            xmltree = reader.metadata.xmltree
            image = reader.read_image()

        image, xmltree = sicd_pixel_type.sicd_as_re32f_im32f(image, xmltree)
        sicdew = sksicd.ElementWrapper(xmltree.getroot())
        assert sicdew["Grid"]["Row"]["Sgn"] == sicdew["Grid"]["Col"]["Sgn"]
        if sicdew["Grid"]["Row"]["Sgn"] == 1:
            image = np.conjugate(image)
            sicdew["Grid"]["Row"]["Sgn"] = sicdew["Grid"]["Col"]["Sgn"] = -1

        image, removed_pwr_frac, removed_data_frac = prf_alias_removal(
            image.astype("complex64"),
            xmltree,
            config.num_iters,
            zones,
            threshold=config.threshold,
            dilate=config.dilate,
            prf=config.prf,
        )
        metadata = reader.metadata
        metadata.xmltree = xmltree

        with (
            open(config.output_sicd_filename, "wb") as file,
            sksicd.NitfWriter(file, reader.metadata) as writer,
        ):
            writer.write_image(image)

        print("removed_pwr_frac = ", removed_pwr_frac)
        print("removed_data_frac = ", removed_data_frac)

        return 0
