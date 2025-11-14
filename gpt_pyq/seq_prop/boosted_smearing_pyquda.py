# boosted_smearing_pyquda.py — streamlined backend handling

from typing import Sequence
from math import pi
from pyquda.field import Ns, Nc
from pyquda.field import LatticeInfo, LatticeGauge, LatticeFermion, LatticePropagator

# ---------- backend picking (generic) ----------
def _get_xp_from_array(a):
    """Return the base module of the array's type, e.g. cupy / numpy / dpnp / torch."""
    base = type(a).__module__.split('.')[0]
    return __import__(base)

def _ensure_backend(x, xp):
    """Move x to the same backend as xp if needed."""
    if type(x).__module__.split('.')[0] == xp.__name__:
        return x
    if hasattr(xp, "asarray"):
        return xp.asarray(x)
    if xp.__name__ == "torch":
        return xp.as_tensor(x)
    return x  # fallback

def _fftnd(xp, a, axes, inverse=False):
    if xp.__name__ == "torch":
        return (xp.fft.ifftn if inverse else xp.fft.fftn)(a, dim=axes)
    return (xp.fft.ifftn if inverse else xp.fft.fftn)(a, axes=axes)

def _exp_complex(xp, real, imag):
    if xp.__name__ == "torch":
        return xp.exp(real) * (xp.cos(imag) + 1j * xp.sin(imag))
    return xp.exp(real + 1j * imag)

# ---------- layout helpers ----------
def _eo_to_full(xp, psi_eo, Lt, Lz, Ly, Lx):
    full = xp.zeros((Lt, Lz, Ly, Lx, Ns, Nc), dtype=psi_eo.dtype)
    tz = xp.arange(Lt)[:, None, None]
    zz = xp.arange(Lz)[None, :, None]
    yy = xp.arange(Ly)[None, None, :]
    parity_tzy = ((tz + zz + yy) & 1)[..., None]
    xh = xp.arange(Lx // 2)[None, None, None, :]
    tt = xp.arange(Lt)[:, None, None, None]
    z2 = xp.arange(Lz)[None, :, None, None]
    y2 = xp.arange(Ly)[None, None, :, None]
    for eo in (0, 1):
        x_par = (eo - parity_tzy) & 1
        x_full = 2 * xh + x_par
        full[tt, z2, y2, x_full, ...] = psi_eo[eo]
    return full

def _full_to_eo(xp, psi_full, Lt, Lz, Ly, Lx):
    eo = xp.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=psi_full.dtype)
    tz = xp.arange(Lt)[:, None, None]
    zz = xp.arange(Lz)[None, :, None]
    yy = xp.arange(Ly)[None, None, :]
    parity_tzy = ((tz + zz + yy) & 1)[..., None]
    xh = xp.arange(Lx // 2)[None, None, None, :]
    tt = xp.arange(Lt)[:, None, None, None]
    z2 = xp.arange(Lz)[None, :, None, None]
    y2 = xp.arange(Ly)[None, None, :, None]
    for eo_par in (0, 1):
        x_par = (eo_par - parity_tzy) & 1
        x_full = 2 * xh + x_par
        eo[eo_par] = psi_full[tt, z2, y2, x_full, ...]
    return eo

def _left_color_mul_full(xp, U_full, psi_full):
    # U_full: (...,3,3), psi_full: (...,Ns,3) -> (...,Ns,3)
    return xp.einsum("...ab,...sb->...sa", U_full, psi_full)

def _build_kernel_realspace_full(xp, Lx, Ly, Lz, w, k):
    rx = xp.arange(Lx, dtype=xp.float64); rx = (rx + Lx/2) % Lx - Lx/2
    ry = xp.arange(Ly, dtype=xp.float64); ry = (ry + Ly/2) % Ly - Ly/2
    rz = xp.arange(Lz, dtype=xp.float64); rz = (rz + Lz/2) % Lz - Lz/2
    kx, ky, kz = k
    real = (-0.5/(w*w)) * (rx[None,None,:]**2 + ry[None,:,None]**2 + rz[:,None,None]**2)
    imag = 2*pi * ((kx/Lx)*rx[None,None,:] + (ky/Ly)*ry[None,:,None] + (kz/Lz)*rz[:,None,None])
    return _exp_complex(xp, real, imag)  # (Lz, Ly, Lx)

# ---------- smear one fermion ----------
def _boosted_smearing_fermion(U_trafo: LatticeGauge, src: LatticeFermion, *, w: float, boost):
    latt_info: LatticeInfo = src.latt_info
    Lx, Ly, Lz, Lt = latt_info.size

    xp = _get_xp_from_array(src.data)
    U_full = _ensure_backend(U_trafo.data, xp)  # (Lt,Lz,Ly,Lx,3,3)

    # 1) eo+x/2 -> full
    psi_full = _eo_to_full(xp, src.data, Lt, Lz, Ly, Lx)  # (Lt,Lz,Ly,Lx,Ns,Nc)

    # 2) multiply U
    psi_gf = _left_color_mul_full(xp, U_full, psi_full)

    # 3) FFT on (Lz, Ly, Lx) axes
    axes = (1, 2, 3)
    psi_p = _fftnd(xp, psi_gf, axes=axes, inverse=False)

    # 4) kernel FFT and multiply
    Kxyz = _build_kernel_realspace_full(xp, Lx, Ly, Lz, w, boost)     # (Lz,Ly,Lx)
    Kp = _fftnd(xp, Kxyz, axes=(0, 1, 2), inverse=False)              # (Lz,Ly,Lx)
    psi_p = psi_p * Kp[None, ..., None, None]

    # 5) inverse FFT
    psi_back = _fftnd(xp, psi_p, axes=axes, inverse=True)

    # 6) multiply U^\dagger
    Udag = xp.swapaxes(xp.conj(U_full), -1, -2)
    psi_out_full = _left_color_mul_full(xp, Udag, psi_back)

    # 7) full -> eo+x/2
    psi_out_eo = _full_to_eo(xp, psi_out_full, Lt, Lz, Ly, Lx)

    out = LatticeFermion(latt_info)
    out.data = psi_out_eo
    return out

# ---------- public API ----------
def boosted_smearing(
    U_trafo: LatticeGauge,
    src, *,
    w: float,
    boost: Sequence[float],
):
    if isinstance(src, LatticeFermion):
        return _boosted_smearing_fermion(U_trafo, src, w=w, boost=boost)
    if isinstance(src, LatticePropagator):
        out = LatticePropagator(src.latt_info)
        for s in range(Ns):
            for c in range(Nc):
                f_sm = _boosted_smearing_fermion(U_trafo, src.getFermion(s, c), w=w, boost=boost)
                out.setFermion(f_sm, s, c)
        return out
    raise TypeError(f"boosted_smearing: unsupported src type: {type(src)}")