"""Equal-time spatial bilinears, independent of QUDA initialization.

Lexicographic propagators have axes t,z,y,x,sink_spin,source_spin,
sink_color,source_color.  The returned grid has axes bx,by,bz,gamma,t.
The displacement points from the active field to the barred field:
C(b)=sum_x phase(x) Tr[B(x) Gamma F(x-b) Gamma_source].
With phase=exp(-ip.x), the positive displacement DFT selects active k
and spectator p-k.  No time transform or trace of the propagator is made.
"""

import numpy as np


def _backend(array):
    if type(array).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


def backward_line(raw, gamma5):
    """gamma5 raw^dagger gamma5, including the color transpose."""
    xp = _backend(raw)
    return xp.einsum('kl,...jlba,ji->...kiab', gamma5, raw.conj(), gamma5,
                     optimize=True)


def spatial_bilinear_grid(backward, forward, gammas, source_gamma, phase=None):
    """All spatial displacements via two FFTs and one 4x4 spin product.

    backward is already gamma5-conjugated (not a raw forward propagator).
    For MPI, inputs must be spatially complete on each time slab. Output
    needs a time gather only. At most one 4x4 color-pair buffer is live;
    no 256-component spatial correlation tensor is materialized.
    """
    xp = _backend(forward)
    if backward.shape != forward.shape or forward.ndim != 8:
        raise ValueError('Expected equal lexicographic t,z,y,x,4,4,3,3 arrays')
    if forward.shape[-4:] != (4, 4, 3, 3):
        raise ValueError('Invalid spin/color axes')
    if source_gamma.shape != (4, 4):
        raise ValueError('source_gamma must be a single 4x4 matrix')
    if phase is not None and phase.shape != forward.shape[:4]:
        raise ValueError('phase must have axes t,z,y,x')
    axes = (1, 2, 3)
    # Tr[B G F J] = sum_(j,i,l,m,a,b) B[j,i,a,b] G[i,l]
    # F[l,m,b,a] J[m,j]. B has rows=source, cols=insertion spin.
    spin_spectrum = xp.zeros(forward.shape[:4] + (4, 4), dtype=forward.dtype)
    reverse = [(-xp.arange(forward.shape[a])) % forward.shape[a] for a in axes]
    for a in range(3):
        for b in range(3):
            left = xp.einsum('mj,...ji->...mi', source_gamma,
                             backward[..., a, b], optimize=True)
            if phase is not None:
                left *= phase[..., None, None]
            left_k = xp.fft.fftn(left, axes=axes)
            # DFT_b(+): sum_b exp(+ik.b) sum_x L(x)F(x-b)
            # = L_fft(-k) F_fft(k). This is also FFT_b(-) of C(x+b).
            for axis, indices in zip(axes, reverse):
                left_k = xp.take(left_k, indices, axis=axis)
            right_k = xp.fft.fftn(forward[..., b, a], axes=axes)
            spin_spectrum += xp.einsum('...mi,...lm->...il', left_k, right_k,
                                       optimize=True)
    # The computed spectrum is DFT(+). FFT/V reconstructs its b grid.
    volume = int(np.prod(forward.shape[1:4]))
    spin_grid = xp.fft.fftn(spin_spectrum, axes=axes) / volume
    grid = xp.einsum('...il,gil->...g', spin_grid, gammas, optimize=True)
    # Tr convention is G[i,l] with spin_grid indexed i,l (no transpose here).
    return xp.transpose(grid, (3, 2, 1, 4, 0))


def displacement_dft(grid, modes):
    """Positive, unnormalized b Fourier sum; axes bx,by,bz,gamma,t."""
    xp = _backend(grid)
    if grid.ndim != 5:
        raise ValueError('grid axes must be bx,by,bz,gamma,t')
    volume = int(np.prod(grid.shape[:3]))
    spectrum = xp.fft.ifftn(grid, axes=(0, 1, 2)) * volume
    return xp.stack([spectrum[tuple(int(k[d]) % grid.shape[d] for d in range(3))]
                     for k in modes])


def hermitian_basis(raw, physical_from_raw, labels):
    """Orthonormal Hermitian basis in the historical label order."""
    physical = np.einsum('ab,bij->aij', physical_from_raw, raw)
    factors = np.array([1j if label.endswith('5') and label != '5'
                        or label.startswith('S') else 1 for label in labels])
    transform = factors[:, None] * physical_from_raw
    basis = factors[:, None, None] * physical
    gram = np.einsum('aij,bji->ab', basis, basis) / 4
    if not np.allclose(basis.conj().transpose(0, 2, 1), basis, atol=1e-12):
        raise ValueError('Non-Hermitian gamma basis')
    if not np.allclose(gram, np.eye(16), atol=1e-12):
        raise ValueError('Non-orthonormal gamma basis')
    return basis, transform


def color_singlet(propagator):
    """(...,spin,color,spin,color) -> (...,spin,spin)."""
    if propagator.shape[-4:] != (4, 3, 4, 3):
        raise ValueError('Propagator must preserve both spin and color indices')
    return np.einsum('...iaja->...ij', propagator) / 3


def equation_rhs(phi, propagator, basis, insertion):
    """Literal note contraction, with ordinary insertion transpose."""
    inverse = np.linalg.inv(propagator)
    kernel = np.einsum('aij,jk,bkl,li->ab', basis, inverse, basis, insertion.T)
    value = np.einsum('a,ab,b->', phi.conj(), kernel, phi) / 48
    return value, kernel, {'condition_number': float(np.linalg.cond(propagator)),
                           'inverse_residual': float(np.linalg.norm(propagator @ inverse - np.eye(4)))}
