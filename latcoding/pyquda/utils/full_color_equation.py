"""The fixed full spin/color quark-state projection relation being tested.

Inputs are ensemble-averaged S and restored Hermitian-basis phi in lattice
units. Averaging inverse matrices per configuration is a different estimator.
"""
import numpy as np


def full_color_rhs(phi,propagator,basis,insertion,*,energy=None):
    phi=np.asarray(phi);H=np.asarray(basis);gamma=np.asarray(insertion)
    S=np.asarray(propagator,dtype=np.complex128)
    if phi.shape!=(16,) or H.shape!=(16,4,4) or gamma.shape!=(4,4):
        raise ValueError('Expected16 amplitudes,16 Hermitian4x4 matrices and one4x4 insertion')
    if S.shape==(4,3,4,3):S=S.reshape(12,12)
    if S.shape!=(12,12):raise ValueError('Full12x12 spin/color propagator is required')
    if not all(np.isfinite(a).all() for a in [phi,H,gamma,S]):raise ValueError('Nonfinite equation input')
    if not np.allclose(H.conj().transpose(0,2,1),H,atol=1e-12):raise ValueError('Basis must be Hermitian')
    if not np.allclose(np.einsum('aij,bji->ab',H,H)/4,np.eye(16),atol=1e-12):
        raise ValueError('Basis must be trace orthonormal')
    singular=np.linalg.svd(S,compute_uv=False)
    if singular[-1]<=0:raise np.linalg.LinAlgError('Singular full propagator')
    M=np.einsum('i,iab->ab',phi,H)/4
    A=np.kron(M/3,np.eye(3));solved=np.linalg.solve(S,A)
    raw=np.trace(A.conj().T@solved@np.kron(gamma.T,np.eye(3)))
    inverse=np.linalg.solve(S,np.eye(12))
    spin=np.einsum('iaja->ij',S.reshape(4,3,4,3))/3
    diagnostics=dict(condition_number=float(singular[0]/singular[-1]),singular_values=singular.tolist(),
        inverse_residual=float(np.linalg.norm(S@inverse-np.eye(12))),
        solve_residual=float(np.linalg.norm(S@solved-A)/max(np.linalg.norm(A),1e-300)),
        color_nonsinglet_relative_l2=float(np.linalg.norm(S-np.kron(spin,np.eye(3)))/np.linalg.norm(S)))
    if energy is None:return raw,diagnostics
    if not np.isfinite(energy) or energy<=0:raise ValueError('Positive lattice-unit energy is required')
    return raw/(2*energy),diagnostics
