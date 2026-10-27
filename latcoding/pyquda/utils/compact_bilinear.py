"""Rank-factorized meson bilinears with bounded spatial FFT buffers.

Compact arrays: (t,z,y,x,spin_sink,color_sink,source_column).
Coordinate output: (t,bz,by,bx,gamma), with active endpoint x-b.
No time transform. Source_column includes projected spin AND source color.
"""
import numpy as np


def backend(array):
    if type(array).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


def source_factors(gamma_source, gamma_time, gamma5, tolerance=1e-12):
    """Factor Gamma_bar_source gamma5 = A D†; Wick minus is explicit.

    Gamma_bar_source=gamma_time Gamma_source† gamma_time.
    Both factors have shape (4,rank), yielding rank*Nc RHS per line.
    """
    def cpu(value):
        return value.get() if hasattr(value, 'get') else np.asarray(value)
    source, time_gamma, chirality = map(cpu, (gamma_source, gamma_time, gamma5))
    bar = time_gamma @ source.conj().T @ time_gamma
    target = bar @ chirality
    u, s, vh = np.linalg.svd(target)
    keep = s > tolerance * s.max()
    a = u[:, keep] * np.sqrt(s[keep])
    d = vh.conj().T[:, keep] * np.sqrt(s[keep])
    np.testing.assert_allclose(a @ d.conj().T, target, atol=tolerance, rtol=tolerance)
    return a, d


def periodic_kernel(shape_xyz, width, boost, *, xp=np):
    """Normalized source convolution kernel, centered periodic coordinates."""
    widths = np.broadcast_to(np.asarray(width, dtype=float), (3,))
    if np.any(widths <= 0) or not np.isfinite(widths).all():
        raise ValueError('Positive finite widths required')
    axes = []
    for n, w, k in zip(shape_xyz, widths, boost):
        r = (xp.arange(n) + n // 2) % n - n // 2
        v = xp.exp(-.5 * (r / w)**2 + 2j * xp.pi * k * r / n)
        axes.append(v / xp.abs(v).sum())
    return axes[2][:, None, None] * axes[1][None, :, None] * axes[0][None, None, :]


def spatial_fft(compact):
    if compact.ndim != 7 or compact.shape[-3:-1] != (4, 3):
        raise ValueError('Expected (t,z,y,x,4,3,ncolumn)')
    return backend(compact).fft.fftn(compact, axes=(1, 2, 3))


def spin_spectrum(active_fft, spectator_fft, *, p=(0, 0, 0), origin=(0, 0, 0),
                  active_sink_kernel=None, spectator_sink_kernel=None):
    """Positive displacement DFT of the Wick-minus compact spin tensor.

    For external exp(-ip.(x-origin)), left mode is conj(D_fft(k-p)).
    Kernels, when provided, describe ADJOINT sink smearing.
    """
    xp = backend(active_fft)
    if active_fft.shape != spectator_fft.shape:
        raise ValueError('Compact lines must have matching axes')
    lengths = active_fft.shape[1:4]
    left = spectator_fft
    if spectator_sink_kernel is not None:
        left = left * spectator_sink_kernel.conj()[None, ..., None, None, None]
    for axis, momentum in zip((1, 2, 3), p[::-1]):
        if momentum:
            left = xp.roll(left, int(momentum), axis=axis)
    right = active_fft
    if active_sink_kernel is not None:
        right = right * active_sink_kernel.conj()[None, ..., None, None, None]
    result = -xp.einsum('tzyxicr,tzyxjcr->tzyxij', left.conj(), right, optimize=True)
    constant = xp.exp(2j*xp.pi*sum(float(n)*float(o)/l for n,o,l in zip(p,origin,lengths[::-1])))
    result *= constant
    return result


def gamma_spectrum(spin, gammas, gamma5):
    xp = backend(spin)
    insertion = xp.einsum('ij,gjk->gik', xp.asarray(gamma5), xp.asarray(gammas))
    return xp.einsum('tzyxij,gij->tzyxg', spin, insertion, optimize=True)


def coordinate_grid(spin, gammas, gamma5):
    xp = backend(spin)
    volume = int(np.prod(spin.shape[1:4]))
    # spin is DFT_b(+), so FFT_b(-)/V reconstructs C(b).
    return gamma_spectrum(xp.fft.fftn(spin, axes=(1,2,3))/volume, gammas, gamma5)


def selected_from_spectrum(spin, modes, gammas, gamma5):
    xp = backend(spin)
    selected = xp.stack([spin[:, int(k[2]) % spin.shape[1],
                              int(k[1]) % spin.shape[2], int(k[0]) % spin.shape[3]]
                         for k in modes], axis=1)
    insertion = xp.einsum('ij,gjk->gik', xp.asarray(gamma5), xp.asarray(gammas))
    return xp.einsum('tkij,gij->tkg', selected, insertion, optimize=True)


def direct_coordinate(active, spectator, b, gammas, gamma5, *, p=(0,0,0), origin=(0,0,0)):
    """Independent explicit shifted-array reference for numerical validation."""
    xp = backend(active)
    shifted = xp.roll(active, tuple(int(v) for v in b[::-1]), axis=(1,2,3))
    z,y,x = xp.indices(active.shape[1:4])
    phase = xp.exp(-2j*xp.pi*sum(n*(c-o)/l for n,c,o,l in
                               zip(p,(x,y,z),origin,active.shape[1:4][::-1])))
    insertion = xp.einsum('ij,gjk->gik', xp.asarray(gamma5), xp.asarray(gammas))
    return -xp.einsum('tzyxicr,gij,tzyxjcr,zyx->tg',spectator.conj(),insertion,shifted,phase,optimize=True)


def selected_direct(active, spectator, modes, gammas, gamma5, *, p=(0,0,0), origin=(0,0,0)):
    """Independent selected-mode sums without an FFT or a coordinate grid."""
    xp = backend(active)
    z,y,x = xp.indices(active.shape[1:4])
    coords=(x,y,z);sizes=active.shape[1:4][::-1]
    insertion=xp.einsum('ij,gjk->gik',xp.asarray(gamma5),xp.asarray(gammas))
    values=[]
    for k in modes:
        right_phase=xp.exp(-2j*xp.pi*sum(n*c/l for n,c,l in zip(k,coords,sizes)))
        left_phase=xp.exp(2j*xp.pi*sum((n-m)*c/l+m*o/l for n,m,c,o,l in zip(k,p,coords,origin,sizes)))
        right=xp.einsum('zyx,tzyxicr->ticr',right_phase,active,optimize=True)
        left=xp.einsum('zyx,tzyxicr->ticr',left_phase,spectator.conj(),optimize=True)
        values.append(-xp.einsum('ticr,gij,tjcr->tg',left,insertion,right,optimize=True))
    return xp.stack(values,axis=1)
