"""Unit-wall S0 and the production magnetic-clover contact, without other RHS."""
import numpy as np
from . import equal_time_matching as m
from . import streamed_wall_propagator as wall
from . import compact_meson_runtime as runtime


def contact_column(links,magnetic,mass,q,col):
    xp=m.backend(links);positive=m.phase([links.shape[3],links.shape[2],links.shape[1]],q,xp)
    rhs=xp.zeros(positive.shape+(4,3),complex);rhs[...,col//3,col%3]=positive.conj()
    operator=lambda f:m.bw(links,f,mass)+m.local_apply(magnetic,f)
    solution,residual,iterations=m.cg_solve(operator,rhs)
    return m.host(m.project(solution,positive)),residual,iterations


def measure(dirac,gauge,cfg,times,qis,tis,gammas,writer,deadline):
    info=dirac.latt_info;new=0
    valid=info.mpi_comm.bcast(writer.valid.copy() if info.mpi_rank==0 else None,root=0)
    contact=info.mpi_comm.bcast(writer.contact_valid.copy() if info.mpi_rank==0 else None,root=0)
    import cupy as cp
    for ti in tis:
        if valid[qis,ti].all() and contact[qis,ti].all():continue
        runtime.budget_check(info,deadline)
        needs_contact=not contact[qis,ti].all();links=magnetic=None
        if needs_contact:
            links=m.gather_slice(info,gauge.data,times[ti],gauge=True)
            if info.mpi_rank==0:links=cp.asarray(links)
            clover=m.actual_clover_slice(dirac,times[ti],links,cp.asarray(gammas))
            if info.mpi_rank==0:
                clover=cp.asarray(clover)
                assert float(cp.linalg.norm(clover-clover.conj().swapaxes(-1,-2))/max(float(cp.linalg.norm(clover)),1.))<=1e-10
                magnetic,_=m.split_clover(clover,cp.asarray(gammas[3]))
            del clover
        for qi in qis:
            for col in range(12):
                if not valid[qi,ti,col]:
                    runtime.budget_check(info,deadline)
                    value,residual=wall.solve_wall_column(dirac,cfg['q_list'][qi],times[ti],col//3,col%3,
                        threshold=cfg['residual_threshold'],deadline=deadline)
                    if info.mpi_rank==0:writer.column('S0',qi,ti,col,value,residual)
                    valid[qi,ti,col]=True;new+=1
                if not contact[qi,ti,col]:
                    runtime.budget_check(info,deadline)
                    if info.mpi_rank==0:
                        value,residual,iterations=contact_column(links,magnetic,cfg['mass'],cfg['q_list'][qi],col)
                        writer.column('A_raw',qi,ti,col,value,residual,iterations)
                    contact[qi,ti,col]=True;new+=1
                cp.get_default_memory_pool().free_all_blocks()
            if info.mpi_rank==0:print(f'UNIT CONTACT BLOCK q={qi} absolute_t={times[ti]}',flush=True)
        del links,magnetic
        cp.get_default_memory_pool().free_all_blocks()
    return new
