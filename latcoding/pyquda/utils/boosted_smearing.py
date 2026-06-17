'''
Modified by Jinchen He, 2025-11-21.
Refactored for MPI support using pyquda_utils.fft by Gemini.
Optimized: Removed U_trafo dependency (Assuming Identity Gauge).

This module implements boosted smearing in PyQUDA.
It supports MPI parallelization by leveraging PyQUDA's internal distributed FFT utils.

momentum smearing: add a momentum phase to the gauge, then do the gaussian smearing;
boosted smearing: apply a gauge-covariant Gaussian convolution with an injected momentum: the source is first rotated into the fixed gauge frame, Fourier-transformed, multiplied by the momentum-shifted Gaussian kernel in momentum space, inverse-transformed back to position space, and finally rotated back with the hermitian conjugate of the gauge transformation.
'''

from typing import Sequence
from math import pi
from time import perf_counter
import numpy as np

# PyQUDA imports
from pyquda.field import Ns, Nc
from pyquda.field import LatticeInfo, LatticeFermion, LatticePropagator, LatticeComplex

# Import PyQUDA's distributed FFT
try:
    from pyquda_utils.fft import fft, ifft
except ImportError:
    raise ImportError("Could not import 'fft' from 'pyquda_utils'. Please ensure PyQUDA is installed correctly.")

from pyquda_comm import getMPIRank, getCoordFromRank
from latcoding.pyquda.utils.tools import _get_xp_from_array, _asarray_on_queue, mpi_timer_print

def _exp_complex(xp, real, imag):
    if xp.__name__ == "torch":
        return xp.exp(real) * (xp.cos(imag) + 1j * xp.sin(imag))
    return xp.exp(real + 1j * imag)

def _get_global_grid_coords(xp, latt_info: LatticeInfo):
    """
    generate the global coordinates of the MPI Rank (Global Coordinates).
    """
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.global_size
    
    rank = getMPIRank()
    coords = getCoordFromRank(rank) 
    
    # calculate the offset of the current rank in the global lattice
    off_t = coords[3] * Lt
    off_z = coords[2] * Lz
    off_y = coords[1] * Ly
    off_x = coords[0] * Lx

    # generate the local coordinates (0 ~ L-1)
    rx_local = xp.arange(Lx, dtype=xp.float64)
    ry_local = xp.arange(Ly, dtype=xp.float64)
    rz_local = xp.arange(Lz, dtype=xp.float64)

    # convert to global coordinates, and handle the periodic boundary conditions (centered at 0)
    rx = (rx_local + off_x + Gx/2) % Gx - Gx/2
    ry = (ry_local + off_y + Gy/2) % Gy - Gy/2
    rz = (rz_local + off_z + Gz/2) % Gz - Gz/2

    return rx, ry, rz

def _build_kernel_realspace_distributed(xp, latt_info: LatticeInfo, w: float, boost: Sequence[float]):
    """
    build the distributed real space Gaussian kernel.
    return a LatticeComplex object (Checkerboard layout), which can be directly passed to fft.
    """
    rx, ry, rz = _get_global_grid_coords(xp, latt_info)
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.global_size
    
    kx, ky, kz = boost

    # Broadcasting: (Lz, Ly, Lx)
    rx = rx[None, None, :]
    ry = ry[None, :, None]
    rz = rz[:, None, None]

    # calculate the exponential part
    real = (-0.5/(w*w)) * (rx**2 + ry**2 + rz**2)
    imag = 2*pi * ((kx/Gx)*rx + (ky/Gy)*ry + (kz/Gz)*rz)
    
    k_xyz = _exp_complex(xp, real, imag) # Shape: (Lz, Ly, Lx)

    # wrap the 3D Kernel into LatticeComplex (4D Field)
    kernel_field = LatticeComplex(latt_info)
    
    # create the local full array (Lt, Lz, Ly, Lx)
    k_full_local = xp.zeros((Lt, Lz, Ly, Lx), dtype=xp.complex128)
    k_full_local[:] = k_xyz[None, ...] # Broadcast time
    
    # explicitly convert to CPU (NumPy) and then call evenodd
    k_full_local_cpu = xp.asnumpy(k_full_local)
    cb_data = latt_info.evenodd(k_full_local_cpu, False)
    
    # assign to field.data on the exact same SYCL queue as the PyQUDA field
    kernel_field.data = _asarray_on_queue(cb_data, xp, kernel_field.data)
    
    return kernel_field

def _boosted_smearing_fermion(src: LatticeFermion, *, w: float, boost: Sequence[float]):
    """
    Core implementation of boosted smearing for a single fermion.
    Optimized: Assumes Identity Gauge (No U_trafo input).
    """
    latt_info: LatticeInfo = src.latt_info
    xp = _get_xp_from_array(src.data)

    # ---------------------------------------------------------
    # 1. Forward FFT (Distributed)
    # ---------------------------------------------------------
    # because U=Identity, so we don't need to do src.lexico() -> einsum -> evenodd()
    # directly do FFT on LatticeFermion
    psi_p = fft(src, fft3d=True, backend="cupy" if xp.__name__=="cupy" else "numpy")

    # ---------------------------------------------------------
    # 2. Apply Momentum Space Kernel
    # ---------------------------------------------------------
    K_xyz = _build_kernel_realspace_distributed(xp, latt_info, w, boost)
    K_p = fft(K_xyz, fft3d=True, backend="cupy" if xp.__name__=="cupy" else "numpy")

    # multiply in momentum space: psi(k) * K(k)
    psi_p.data = psi_p.data * K_p.data[..., None, None]

    # ---------------------------------------------------------
    # 3. Inverse FFT (Distributed)
    # ---------------------------------------------------------
    psi_smeared = ifft(psi_p, fft3d=True, backend="cupy" if xp.__name__=="cupy" else "numpy")

    # ---------------------------------------------------------
    # 4. Result
    # ---------------------------------------------------------
    # because U=Identity, we don't need to do the inverse Gauge Rotation
    # psi_smeared is already the final result
    
    return psi_smeared

# ---------- public API ----------
def boosted_smearing(
    src,
    *,
    w: float,
    boost: Sequence[float],
):
    t0 = perf_counter()
    if isinstance(src, LatticeFermion):
        out = _boosted_smearing_fermion(src, w=w, boost=boost)
        mpi_timer_print(src.latt_info, "boosted_smearing", perf_counter() - t0, field="LatticeFermion", width=w, boost=boost)
        return out
    if isinstance(src, LatticePropagator):
        out = LatticePropagator(src.latt_info)
        for s in range(Ns):
            for c in range(Nc):
                # pass in a single fermion
                f_sm = _boosted_smearing_fermion(src.getFermion(s, c), w=w, boost=boost)
                out.setFermion(f_sm, s, c)
        mpi_timer_print(src.latt_info, "boosted_smearing", perf_counter() - t0, field="LatticePropagator", width=w, boost=boost)
        return out
    raise TypeError(f"boosted_smearing: unsupported src type: {type(src)}")
