import numpy as np
from pyquda_utils import gamma
from latcoding.pyquda.utils.fermion_bilinear_basis import (
    GAMMA_LABELS, PYQUDA_GAMMA_IDS, PHYSICAL_FROM_PYQUDA,
)
from latcoding.pyquda.utils.momentum_space import (
    backward_line, spatial_bilinear_grid, displacement_dft,
    hermitian_basis, equation_rhs, color_singlet,
)


def matrices():
    values = [gamma.gamma(i) for i in PYQUDA_GAMMA_IDS]
    return np.stack([np.asarray(v.matrix if hasattr(v, 'matrix') else v) for v in values])


def test_spatial_grid_against_direct_spin_color_trace():
    rng = np.random.default_rng(2)
    shape = (2, 4, 4, 4, 4, 4, 3, 3)
    forward, raw = [rng.normal(size=shape) + 1j*rng.normal(size=shape) for _ in range(2)]
    gs = matrices()
    back = backward_line(raw, gs[0])
    phase = np.exp(-2j*np.pi*np.indices(shape[:4])[1]/4)
    source = gs[2]
    actual = spatial_bilinear_grid(back, forward, gs, source, phase)
    for bx, by, bz in [(0,0,0), (1,2,3), (3,1,2), (2,2,2)]:
        shifted = np.roll(forward, (bz, by, bx), axis=(1,2,3))
        expected = np.einsum('tzyxjiab,gil,tzyxlmba,mj,tzyx->gt',
                             back, gs, shifted, source, phase, optimize=True)
        np.testing.assert_allclose(actual[bx,by,bz], expected, rtol=1e-10, atol=1e-10)
    modes = [(1,2,3), (2,0,1)]
    transformed = displacement_dft(actual, modes)
    x,y,z = np.indices((4,4,4))
    for idx,k in enumerate(modes):
        w = np.exp(2j*np.pi*(k[0]*x+k[1]*y+k[2]*z)/4)
        np.testing.assert_allclose(transformed[idx],
                                  np.einsum('xyz,xyzgt->gt',w,actual), atol=1e-9)


def test_plane_waves_select_k_and_p_minus_k():
    n = 4
    z,y,x = np.indices((n,n,n))
    k = np.array([1,1,0]); p = np.array([0,0,1]); q = p-k
    coords = np.stack([x,y,z])
    forward_phase = np.exp(2j*np.pi*np.einsum('i,izyx->zyx',k,coords)/n)
    raw_phase = np.exp(-2j*np.pi*np.einsum('i,izyx->zyx',q,coords)/n)
    phase = np.exp(-2j*np.pi*np.einsum('i,izyx->zyx',p,coords)/n)[None]
    identity = np.einsum('ij,ab->ijab',np.eye(4),np.eye(3))
    forward = forward_phase[None,...,None,None,None,None]*identity
    raw = raw_phase[None,...,None,None,None,None]*identity
    gs=matrices()
    grid=spatial_bilinear_grid(backward_line(raw,gs[0]),forward,np.eye(4)[None],np.eye(4),phase)
    result=displacement_dft(grid,[k, -k])[:,0,0]
    np.testing.assert_allclose(result,[12*n**6,0],atol=1e-8)


def test_basis_reconstruction_and_transposed_equation():
    raw=matrices(); basis,transform=hermitian_basis(raw,PHYSICAL_FROM_PYQUDA,GAMMA_LABELS)
    rng=np.random.default_rng(1)
    m=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    raw_phi=np.einsum('ij,aji->a',m,raw)
    phi=transform@raw_phi
    reconstructed=np.einsum('a,aij->ij',phi,basis)/4
    np.testing.assert_allclose(reconstructed,m,atol=1e-12)
    s=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))+4*np.eye(4)
    insertion=raw[GAMMA_LABELS.index('Y')]
    actual,kernel,diag=equation_rhs(phi,s,basis,insertion)
    expected=np.trace(m.conj().T@np.linalg.inv(s)@m@insertion.T)/3
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert diag['inverse_residual']<1e-12
    full=np.einsum('ij,ab->iajb',s,np.eye(3))
    np.testing.assert_allclose(color_singlet(full),s)
