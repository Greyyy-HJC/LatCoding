"""Normalized boosted smearing with bounded Fourier-kernel reuse.

The historical boosted_smearing API and its unnormalized behavior are intact.
The new entrypoint normalizes sum_r |K(r)|=1. Cached kernels are never mutated.
"""
from collections import OrderedDict
import numpy as np

_KERNELS = OrderedDict()


def normalized_boosted_smearing(src, *, width, boost, adjoint=False):
    from pyquda_utils.fft import fft, ifft
    from latcoding.pyquda.utils.boosted_smearing import _build_kernel_realspace_distributed
    from latcoding.pyquda.utils.tools import _get_xp_from_array
    widths = np.broadcast_to(np.asarray(width, dtype=float), (3,))
    if np.all(widths == 0):
        return src
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError('Smearing widths must all be positive, or all zero')
    boost = tuple(float(v) for v in boost)
    if len(boost) != 3 or not np.all(np.isfinite(boost)):
        raise ValueError('Expected three finite boost modes')
    xp = _get_xp_from_array(src.data)
    info = src.latt_info
    key = (tuple(info.global_size), tuple(info.size), info.mpi_rank,
           tuple(widths), boost, xp.__name__)
    backend = 'cupy' if xp.__name__ == 'cupy' else 'numpy'
    if key not in _KERNELS:
        kernel = _build_kernel_realspace_distributed(xp, info, widths, boost)
        normalizer = 1.0
        for length, w in zip(info.global_size[:3], widths):
            r = (np.arange(length) + length / 2) % length - length / 2
            normalizer *= np.exp(-0.5 * (r / w)**2).sum()
        kernel.data /= normalizer
        _KERNELS[key] = fft(kernel, fft3d=True, backend=backend)
        while len(_KERNELS) > 2:
            _KERNELS.popitem(last=False)
    _KERNELS.move_to_end(key)
    field = fft(src, fft3d=True, backend=backend)
    kernel_data=_KERNELS[key].data.conj() if adjoint else _KERNELS[key].data
    field.data *= kernel_data[(...,) + (None,) * len(src.field_shape)]
    return ifft(field, fft3d=True, backend=backend)
