"""Stream complete compact-meson coordinate correlators by global time slab."""
import numpy as np
from .compact_bilinear import (
    backend,periodic_kernel,spatial_fft,spin_spectrum,coordinate_grid,
    selected_from_spectrum,direct_coordinate,selected_direct,
)
from .compact_meson_runtime import host,budget_check


def stream_bilinear(active,spectator,slabs,*,p_list,modes,gammas,gamma5,origin,
                    wanted,callback,deadline,time_block=1,external_p=True,
                    pion_gamma=None,width=None,active_boost=None,spectator_boost=None,
                    validate=False):
    """All ranks participate; callback on world rank0 receives bounded chunks.

    wanted axes (p,source_relative_t); C_b payload axes (t,bz,by,bx,gamma).
    p_list labels the pion; PDF's sequential line already projects it, so
    external_p=False. Local SP/SS 2pt can be returned with the same FFTs.
    """
    info=active.latt_info
    xp=backend(active.data)
    checked=False
    for start in range(0,info.Lt,time_block):
        budget_check(info,deadline)
        stop=min(start+time_block,info.Lt)
        absolute=np.arange(start,stop)+info.gt*info.Lt
        relative=(absolute-origin[3])%info.GLt
        wanted_p=[i for i in range(len(p_list)) if np.any(wanted[i,relative[relative<wanted.shape[1]]])]
        payload=None
        if wanted_p:
            ah=slabs.gather(active,start,stop)
            dh=slabs.gather(spectator,start,stop)
            if slabs.key==0:
                af=xp.asarray(ah);df=xp.asarray(dh);del ah,dh
                ak=spatial_fft(af);dk=spatial_fft(df)
                if not validate or checked:
                    del af,df
                ka=kd=None
                if pion_gamma is not None:
                    ka=xp.fft.fftn(periodic_kernel(info.global_size[:3],width,active_boost,xp=xp))
                    kd=xp.fft.fftn(periodic_kernel(info.global_size[:3],width,spectator_boost,xp=xp))
                payload=[]
                for pi in wanted_p:
                    pp=p_list[pi] if external_p else (0,0,0)
                    oo=origin[:3] if external_p else (0,0,0)
                    spin=spin_spectrum(ak,dk,p=pp,origin=oo)
                    grid=coordinate_grid(spin,gammas,gamma5)
                    selected=selected_from_spectrum(spin,modes,gammas,gamma5)
                    error=None
                    if validate and not checked:
                        refs=selected_direct(af,df,modes,gammas,gamma5,p=pp,origin=oo)
                        np.testing.assert_allclose(host(selected),host(refs),rtol=1e-8,atol=1e-10)
                        max_absolute=float(xp.max(xp.abs(selected-refs)))
                        relative_error=float(xp.linalg.norm(selected-refs)/max(float(xp.linalg.norm(refs)),1e-30))
                        for b in [(0,0,0),(1,2,3),(info.GLx-1,7%info.GLy,9%info.GLz)]:
                            ref=direct_coordinate(af,df,b,gammas,gamma5,p=pp,origin=oo)
                            observed=grid[:,b[2],b[1],b[0]]
                            np.testing.assert_allclose(host(observed),host(ref),rtol=1e-8,atol=1e-10)
                            max_absolute=max(max_absolute,float(xp.max(xp.abs(observed-ref))))
                        # Explicit full C_b->DFT check, independent of the stored selected modes.
                        bfft=xp.fft.ifftn(grid,axes=(1,2,3))*info.GLx*info.GLy*info.GLz
                        projected=xp.stack([bfft[:,k[2]%info.GLz,k[1]%info.GLy,k[0]%info.GLx] for k in modes],axis=1)
                        np.testing.assert_allclose(host(selected),host(projected),rtol=1e-8,atol=1e-10)
                        error=dict(max_absolute=max_absolute,relative_norm=relative_error)
                        del refs,bfft,projected
                    sp=ss=None
                    if pion_gamma is not None:
                        insertion=xp.asarray(gamma5)@xp.asarray(pion_gamma)
                        volume=info.GLx*info.GLy*info.GLz
                        sp=host(xp.einsum('tij,ij->t',spin.sum(axis=(1,2,3))/volume,insertion))
                        spin_ss=spin_spectrum(ak,dk,p=pp,origin=oo,active_sink_kernel=ka,spectator_sink_kernel=kd)
                        ss=host(xp.einsum('tij,ij->t',spin_ss.sum(axis=(1,2,3))/volume,insertion))
                        del spin_ss
                    payload.append(dict(p_index=pi,absolute_t=absolute,relative_t=relative,
                        C_b=host(grid),C_k=host(selected),C2_SP=sp,C2_SS=ss,validation=error))
                    del spin,grid,selected
                del ak,dk,ka,kd
                if validate and not checked:
                    del af,df
                    checked=True
        slabs.collect_payload(payload,lambda values:[callback(v) for v in values])
