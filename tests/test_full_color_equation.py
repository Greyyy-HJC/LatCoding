import numpy as np
from latcoding.pyquda.utils.full_color_equation import full_color_rhs


def basis():
    I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
    return np.array([np.kron(a,b) for a in [I,X,Y,Z] for b in [I,X,Y,Z]])


def test_full_color_trace_matches_literal256_terms_and_ordinary_transpose():
    rng=np.random.default_rng(99);H=basis();phi=rng.normal(size=16)+1j*rng.normal(size=16)
    S=np.eye(12)+.07*(rng.normal(size=(12,12))+1j*rng.normal(size=(12,12)))
    G=H[6]+.1j*H[10] # Nonsymmetric/nonreal insertion distinguishes .T from dagger.
    value,diagnostics=full_color_rhs(phi,S,H,G,energy=.7)
    inv=np.linalg.inv(S);expected=0j
    for i in range(16):
        for j in range(16):
            term=np.trace(np.kron(H[i],np.eye(3))@inv@np.kron(H[j],np.eye(3))@np.kron(G.T,np.eye(3)))
            expected+=phi[i].conj()*phi[j]*term/(16*3**2*2*.7)
    np.testing.assert_allclose(value,expected,rtol=1e-13,atol=1e-13)
    assert diagnostics['solve_residual']<1e-13 and diagnostics['inverse_residual']<1e-13


def test_full_color_reduces_to_spin_formula_only_with_color_identity():
    rng=np.random.default_rng(98);H=basis();phi=rng.normal(size=16)+1j*rng.normal(size=16)
    spin=np.eye(4)+.08*(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    G=H[9];S=np.kron(spin,np.eye(3))
    value,diagnostics=full_color_rhs(phi,S,H,G)
    M=np.einsum('i,iab->ab',phi,H)/4
    expected=np.trace(M.conj().T@np.linalg.solve(spin,M)@G.T)/3
    np.testing.assert_allclose(value,expected,rtol=1e-13,atol=1e-13)
    assert diagnostics['color_nonsinglet_relative_l2']<1e-14
