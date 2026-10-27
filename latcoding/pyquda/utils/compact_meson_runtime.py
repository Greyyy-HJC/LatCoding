"""Column-streamed solves and time-block spatial gathers for compact mesons.

This module leaves the historical full-propagator measurement APIs intact.
All inversion residuals use the unpreconditioned Dirac operator independently.
"""
import time
import numpy as np
from .compact_bilinear import backend,periodic_kernel,spatial_fft


class BudgetExhausted(RuntimeError):
    pass


def budget_check(info, deadline):
    expired = info.mpi_comm.bcast(time.monotonic() >= deadline if info.mpi_rank == 0 else None, root=0)
    if expired:
        raise BudgetExhausted('Reached the collective-safe measurement budget')


def host(array):
    return array.get() if hasattr(array,'get') else np.asarray(array)


def block_lexico(columns, start, stop):
    """Convert only selected local time slices, including their parity offset."""
    cb=host(columns.data[:,:,start:stop])
    nr,_,nt,nz,ny,nxh,ns,nc=cb.shape
    lex=np.empty((nr,nt,nz,ny,2*nxh,ns,nc),dtype=cb.dtype)
    t,z,y,x=np.indices((nt,nz,ny,nxh))
    for eo in (0,1):
        lex[:,t,z,y,2*x+((eo-t-start-z-y)%2)]=cb[:,eo]
    return np.ascontiguousarray(lex.transpose(1,2,3,4,5,6,0))


def set_timeslice(rhs, value, local_time):
    """Assign a local (z,y,x,spin,color) slice to a checkerboard fermion."""
    xp=backend(rhs.data)
    nz,ny,nx=value.shape[:3]
    z,y,x=np.indices((nz,ny,nx//2))
    for eo in (0,1):
        cb=value[z,y,2*x+((eo-local_time-z-y)%2)]
        rhs.data[eo,local_time]=xp.asarray(cb)


class SpatialSlabs:
    """One spatial root per time slab; no full 4D propagator gather."""
    def __init__(self,info):
        from mpi4py import MPI
        self.info=info
        self.world=info.mpi_comm
        gx,gy,gz,gt=info.grid_coord
        Gx,Gy,Gz,Gt=info.grid_size
        self.key=gx+Gx*(gy+Gy*gz)
        self.spatial=self.world.Split(gt,self.key)
        self.coords=self.spatial.allgather((gx,gy,gz))
        self.leaders=self.world.Split(0 if self.key==0 else MPI.UNDEFINED,gt)

    def gather(self,columns,start,stop):
        blocks=self.spatial.gather(block_lexico(columns,start,stop),root=0)
        if self.key!=0:
            return None
        nt,nz,ny,nx,ns,nc,nr=blocks[0].shape
        Gx,Gy,Gz,_=self.info.grid_size
        out=np.empty((nt,nz*Gz,ny*Gy,nx*Gx,ns,nc,nr),dtype=blocks[0].dtype)
        for block,(gx,gy,gz) in zip(blocks,self.coords):
            out[:,gz*nz:(gz+1)*nz,gy*ny:(gy+1)*ny,gx*nx:(gx+1)*nx]=block
        return out

    def scatter_slice(self,complete):
        blocks=None
        if self.key==0:
            nx,ny,nz,_=self.info.size
            blocks=[np.ascontiguousarray(complete[gz*nz:(gz+1)*nz,gy*ny:(gy+1)*ny,gx*nx:(gx+1)*nx]) for gx,gy,gz in self.coords]
        return self.spatial.scatter(blocks,root=0)

    def collect_payload(self,value,callback):
        if self.key==0:
            values=self.leaders.gather(value,root=0)
            if self.info.mpi_rank==0:
                for payload in values:
                    if payload is not None:
                        callback(payload)

    def close(self):
        from mpi4py import MPI
        self.spatial.Free()
        if self.leaders!=MPI.COMM_NULL:
            self.leaders.Free()


def invert_column(dirac,rhs,*,threshold,identity,deadline):
    info=dirac.latt_info
    budget_check(info,deadline)
    solution=dirac.invert(rhs)
    applied=dirac.mat(solution)
    difference=rhs-applied
    norm=rhs.norm2()
    residual=float(np.sqrt(difference.norm2()/norm)) if norm>0 else float('nan')
    del applied,difference
    if not np.isfinite(residual) or residual>threshold:
        raise RuntimeError(f'Independent residual failed {identity}: {residual} > {threshold}')
    if info.mpi_rank==0:
        print(f'RHS verified {identity} residual={residual:.6e}',flush=True)
    return solution,residual


def solve_point_columns(dirac,origin,factor,*,width,boost,threshold,deadline):
    """Analytic Gaussian point source; only rank*Nc nonzero RHS are made."""
    from pyquda.field import LatticeFermion,MultiLatticeFermion
    info=dirac.latt_info
    result=MultiLatticeFermion(info,factor.shape[1]*3)
    xp=backend(result.data)
    kernel=periodic_kernel(info.global_size[:3],width,boost,xp=xp)
    nz,ny,nxh=info.Lz,info.Ly,info.Lx//2
    z,y,x=xp.indices((nz,ny,nxh))
    tlocal=origin[3]-info.gt*info.Lt
    residuals=[]
    for r in range(factor.shape[1]):
        for color in range(3):
            rhs=LatticeFermion(info)
            if 0<=tlocal<info.Lt:
                for eo in (0,1):
                    xx=(2*x+((eo-tlocal-z-y)%2)+info.gx*info.Lx-origin[0])%info.GLx
                    yy=(y+info.gy*info.Ly-origin[1])%info.GLy
                    zz=(z+info.gz*info.Lz-origin[2])%info.GLz
                    value=kernel[zz,yy,xx]
                    rhs.data[eo,tlocal,...,color]=value[...,None]*xp.asarray(factor[:,r])
            column=r*3+color
            solution,residual=invert_column(dirac,rhs,threshold=threshold,
                identity=f'point boost={boost} col={column}',deadline=deadline)
            result.data[column]=solution.data
            residuals.append(residual)
            del rhs,solution
    return result,np.asarray(residuals)


def solve_sequential_columns(dirac,spectator,slabs,origin,p,tsep,sink_gamma,gamma5,
                             *,width,active_boost,spectator_boost,threshold,deadline):
    """Build sink-only compact sequential RHS, smear it, solve one column at a time."""
    from pyquda.field import LatticeFermion,MultiLatticeFermion
    info=dirac.latt_info
    target=(origin[3]+tsep)%info.GLt
    tlocal=target-info.gt*info.Lt
    owns=0<=tlocal<info.Lt
    complete=None
    if owns:
        gathered=slabs.gather(spectator,tlocal,tlocal+1)
        if slabs.key==0:
            xp=backend(spectator.data)
            spectral=spatial_fft(xp.asarray(gathered))
            del gathered
            kd=xp.fft.fftn(periodic_kernel(info.global_size[:3],width,spectator_boost,xp=xp))
            spectral*=kd.conj()[None,...,None,None,None]
            smeared=xp.fft.ifftn(spectral,axes=(1,2,3))
            del spectral,kd
            gs=xp.asarray(gamma5)@xp.asarray(sink_gamma).conj().T@xp.asarray(gamma5)
            source=xp.einsum('ij,tzyxjcr->tzyxicr',gs,smeared,optimize=True)
            del smeared
            z,y,x=xp.indices(source.shape[1:4])
            phase=xp.exp(2j*xp.pi*sum(n*(c-o)/l for n,c,o,l in
                                    zip(p,(x,y,z),origin[:3],info.global_size[:3])))
            source*=phase[None,...,None,None,None]
            ka=xp.fft.fftn(periodic_kernel(info.global_size[:3],width,active_boost,xp=xp))
            source=xp.fft.ifftn(spatial_fft(source)*ka[None,...,None,None,None],axes=(1,2,3))
            complete=host(source[0]);del source,ka,phase
    result=MultiLatticeFermion(info,spectator.L5)
    residuals=[]
    for column in range(spectator.L5):
        rhs=LatticeFermion(info)
        if owns:
            local=slabs.scatter_slice(complete[...,column] if slabs.key==0 else None)
            set_timeslice(rhs,local,tlocal)
            del local
        solution,residual=invert_column(dirac,rhs,threshold=threshold,
            identity=f'sequential p={p} tsep={tsep} col={column}',deadline=deadline)
        result.data[column]=solution.data
        residuals.append(residual)
        del rhs,solution
    return result,np.asarray(residuals)
