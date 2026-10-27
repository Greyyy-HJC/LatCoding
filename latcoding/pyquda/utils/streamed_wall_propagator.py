"""Full spin/color equal-time propagator, retaining one solved column.

S(q;t,t) = sum_xy exp(+iq.(x-y)) S(x,t;y,t) / V. Source phase is
negative and sink phase positive. Gaussian smearing and meson source
factorization are not part of this observable.
"""
import numpy as np
from .compact_bilinear import backend
from .compact_meson_runtime import host,invert_column


def wall_phase(local_size,global_size,grid_coord,momentum,local_time,*,xp=np):
    """Positive phase on a checkerboard time slice, axes eo,z,y,xh."""
    nx,ny,nz=local_size[:3]
    if nx%2:raise ValueError('Checkerboard x extent must be even')
    z,y,x=xp.indices((nz,ny,nx//2))
    result=[]
    for eo in (0,1):
        xx=2*x+((eo-local_time-z-y)%2)+grid_coord[0]*nx
        yy=y+grid_coord[1]*ny;zz=z+grid_coord[2]*nz
        angle=2*xp.pi*(momentum[0]*xx/global_size[0]+momentum[1]*yy/global_size[1]
                       +momentum[2]*zz/global_size[2])
        result.append(xp.exp(1j*angle))
    return xp.stack(result)


def solve_wall_column(dirac,momentum,absolute_time,source_spin,source_color,*,threshold,deadline):
    """Collectively solve and project one full-basis column; root gets4x3."""
    from pyquda.field import LatticeFermion
    from mpi4py import MPI
    info=dirac.latt_info
    if not 0<=absolute_time<info.GLt:raise ValueError('Absolute time outside lattice')
    if source_spin not in range(4) or source_color not in range(3):
        raise ValueError('Source spin/color outside full basis')
    rhs=LatticeFermion(info)
    xp=backend(rhs.data)
    local_time=absolute_time-info.gt*info.Lt
    owns=0<=local_time<info.Lt
    phase=None
    if owns:
        phase=wall_phase(info.size,info.global_size,info.grid_coord,momentum,local_time,xp=xp)
        rhs.data[:,local_time,...,source_spin,source_color]=phase.conj()
    solution,residual=invert_column(dirac,rhs,threshold=threshold,deadline=deadline,
        identity=f'wall q={tuple(momentum)} absolute_t={absolute_time} spin={source_spin} color={source_color}')
    if owns:
        projected=xp.einsum('wzyx,wzyxic->ic',phase,solution.data[:,local_time],optimize=True)
        local=host(projected)
    else:local=np.zeros((4,3),dtype=np.complex128)
    output=np.empty((4,3),dtype=np.complex128) if info.mpi_rank==0 else None
    info.mpi_comm.Reduce(np.ascontiguousarray(local),output,op=MPI.SUM,root=0)
    if output is not None:
        output/=int(np.prod(info.global_size[:3]))
        if not np.isfinite(output).all():raise ValueError('Nonfinite projected wall propagator')
    del rhs,solution,phase
    return output,residual
