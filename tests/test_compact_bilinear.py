import numpy as np
from pyquda_utils import gamma
from latcoding.pyquda.utils.fermion_bilinear_basis import PYQUDA_GAMMA_IDS
from latcoding.pyquda.utils.compact_bilinear import (
    source_factors,periodic_kernel,spatial_fft,spin_spectrum,coordinate_grid,
    selected_from_spectrum,selected_direct,direct_coordinate,
)
from latcoding.pyquda.utils.momentum_space import backward_line,spatial_bilinear_grid


def matrices():
    return np.stack([np.asarray(gamma.gamma(i)) for i in PYQUDA_GAMMA_IDS])


def compressed(full,factor):
    a=np.einsum('tzyxijab,jr->tzyxiarb',full,factor)
    return a.reshape(*a.shape[:6],-1)


def test_rank_two_matches_full_spin_color_and_nonzero_phase():
    gs=matrices();g5,g4,g3=gs[0],gs[1],gs[7]
    pion=(g4+1j*g3)@g5/np.sqrt(2)
    a,d=source_factors(pion,g4,g5)
    assert a.shape==d.shape==(4,2)
    rng=np.random.default_rng(812)
    shape=(2,4,4,4,4,4,3,3)
    f,g=[rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in range(2)]
    af,df=compressed(f,a),compressed(g,d)
    p=(0,1,2);origin=(1,2,3)
    z,y,x=np.indices(shape[1:4])
    phase=np.broadcast_to(np.exp(-2j*np.pi*sum(n*(c-o)/4 for n,c,o in zip(p,(x,y,z),origin)))[None],shape[:4])
    source=-(g4@pion.conj().T@g4)
    reference=spatial_bilinear_grid(backward_line(g,g5),f,gs,source,phase)
    spin=spin_spectrum(spatial_fft(af),spatial_fft(df),p=p,origin=origin)
    result=coordinate_grid(spin,gs,g5)
    np.testing.assert_allclose(result,np.transpose(reference,(4,2,1,0,3)),rtol=1e-11,atol=1e-9)
    for b in [(0,0,0),(1,2,3),(3,3,1)]:
        np.testing.assert_allclose(result[:,b[2],b[1],b[0]],direct_coordinate(af,df,b,gs,g5,p=p,origin=origin),atol=1e-9)
    modes=[(1,2,3),(2,0,1),(-1,0,2)]
    selected=selected_from_spectrum(spin,modes,gs,g5)
    np.testing.assert_allclose(selected,selected_direct(af,df,modes,gs,g5,p=p,origin=origin),rtol=1e-11,atol=1e-8)
    spectrum=np.fft.ifftn(result,axes=(1,2,3))*4**3
    np.testing.assert_allclose(selected,np.stack([spectrum[:,k[2]%4,k[1]%4,k[0]%4] for k in modes],axis=1),atol=1e-8)


def test_adjoint_sink_smearing_parseval_matches_explicit_smearing():
    rng=np.random.default_rng(7)
    shape=(1,4,4,4,4,3,6)
    f,d=[rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in range(2)]
    kf=np.fft.fftn(periodic_kernel((4,4,4),(1.1,.7,.9),(0,0,.6)))
    kd=np.fft.fftn(periodic_kernel((4,4,4),(1.1,.7,.9),(0,0,-.6)))
    ff,df=spatial_fft(f),spatial_fft(d)
    fs=np.fft.ifftn(ff*kf.conj()[None,...,None,None,None],axes=(1,2,3))
    ds=np.fft.ifftn(df*kd.conj()[None,...,None,None,None],axes=(1,2,3))
    gs=matrices();p=(0,0,1)
    spin=spin_spectrum(ff,df,p=p,active_sink_kernel=kf,spectator_sink_kernel=kd)
    local=coordinate_grid(spin,gs,gs[0])[:,0,0,0]
    np.testing.assert_allclose(local,direct_coordinate(fs,ds,(0,0,0),gs,gs[0],p=p),atol=1e-10)


def test_normalized_anisotropic_kernel_and_integer_plane_wave():
    k=periodic_kernel((8,8,8),(2.,2.,1.),(0,0,2))
    np.testing.assert_allclose(abs(k).sum(),1,atol=1e-15)
    n=4;z,y,x=np.indices((n,n,n));mom=(1,1,0);p=(0,0,1);q=np.subtract(p,mom)
    fp=np.exp(2j*np.pi*(mom[0]*x+mom[1]*y+mom[2]*z)/n)
    dp=np.exp(-2j*np.pi*(q[0]*x+q[1]*y+q[2]*z)/n)
    f=np.zeros((1,n,n,n,4,3,1),complex);d=f.copy()
    f[...,0,0,0]=fp[None];d[...,0,0,0]=dp[None]
    # Choose gamma so gamma5*Gamma=identity.
    gs=matrices();spin=spin_spectrum(spatial_fft(f),spatial_fft(d),p=p)
    actual=selected_from_spectrum(spin,[mom,(-1,-1,0)],gs[:1],gs[0])
    np.testing.assert_allclose(actual[0,:,0],[-n**6,0],atol=1e-10)
