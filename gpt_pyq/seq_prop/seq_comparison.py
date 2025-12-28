import gpt as g
import numpy as np
from check_funcs import *
from boosted_smearing_pyquda import boosted_smearing

import cupy as cp
from pyquda import init
from pyquda_utils import core, io, gamma, source, gpt
from pyquda_utils.phase import MomentumPhase
from pyquda.field import evenodd
from opt_einsum import contract
from types import SimpleNamespace

width = 2.0
boost_out = [1,2,1]
pf = [1,2,3,2] # [px, py, pz, pt]
t_insert = 4
GEN_SIMD_WIDTH = 64

lat = g.grid([8,8,8,8], g.double)
init([1, 1, 1, 1], resource_path=".cache")


def create_bw_seq_gpt(prop, trafo, origin):
    tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
    
    prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=width, boost=boost_out)
    
    pp = 2.0 * np.pi * np.array(pf) / prop.grid.fdimensions
    P = g.exp_ixp(pp, origin)

    src_seq = up_quark_insertion_gpt(prop, prop, Cg5_gpt, PpSzp_gpt)
    
    # sequential solve through t=t_insert
    src_seq_t = g.lattice(src_seq)
    src_seq_t[:] = 0
    src_seq_t[:, :, :, (origin[3]+t_insert)%prop.grid.fdimensions[3]] = src_seq[:, :, :, (origin[3]+t_insert)%prop.grid.fdimensions[3]]
    
    smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))
        
    tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=width, boost=boost_out)
    
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-8, "maxiter": 1000})

    slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))
    
    tmp_inv = g.mspincolor(prop.grid)
    tmp_inv @= slv * tmp_prop
    
    dst_seq = g.eval(g.adj(tmp_inv) * g.gamma[5])
    
    dst_seq = gpt.LatticePropagatorGPT(dst_seq, GEN_SIMD_WIDTH)
    
    return dst_seq



def create_bw_seq_pyquda(prop, trafo, origin):
    """
    PyQUDA 版构建后向顺序源 (Backward Sequential Source).
    
    参数:
        prop: LatticePropagator
            输入的前向传播子 (u quark)。
        origin: list/tuple
            源坐标，假设格式为 [x, y, z, t] (GPT 风格)。
        t_insert: int
            相对于源时间的插入时间偏移量 (t_sink = (t_source + t_insert) % Lt)。
        pf: list/tuple
            动量，格式为 [px, py, pz, pt] (即 x, y, z, t)。
        Cg5, PpSzp: matrix/int
            用于 up_quark_insertion 的矩阵。
        latt_info: LatticeInfo
            格点信息对象。
            
    返回:
        smearing_input: LatticePropagator
            用于反演的顺序源。
    """
    
    latt_info = prop.latt_info
    Lt = latt_info.Lt
    
    prop = boosted_smearing(trafo, prop, w=width, boost=boost_out)

    # --- 1. 执行重子缩并 (Up Quark Insertion) ---
    src_seq = up_quark_insertion_pyquda(prop, prop, Cg5_pyquda, PpSzp_pyquda)
    
    seq_data = src_seq.data
    
    # --- 2. 时间切片 (Time Slicing) ---
    # origin 格式 [x, y, z, t] -> 时间在 index 3
    t_source = origin[3] 
    t_sink = (t_source + t_insert) % Lt
    
    # 获取数据副本 (GPU)
    seq_data = src_seq.lexico()
    
    # 置零非插入时间片
    mask = np.zeros_like(seq_data)
    mask[t_sink, :, :, :, :, :, :, :] = 1
    seq_data *= mask
    
    seq_data = evenodd(seq_data, axes=[0,1,2,3])
    
    print(">>> shape of seq_data: ", np.shape(seq_data))
    
    # --- 3. 创建动量相位 (Momentum Phase) ---

    # 生成相位
    mom_phase = MomentumPhase(latt_info).getPhase(pf[:3], x0=origin)

    # phase_data = mom_phase_obj.data
    G5 = gamma.gamma(15)
    
    data = contract("ij, wtzyx, wtzyxkjba -> wtzyxikab", G5, mom_phase, seq_data.conj())
    
    smearing_input = core.LatticePropagator(latt_info)
    smearing_input.data = data
    
    src = boosted_smearing(trafo, smearing_input, w=width, boost=boost_out)
    prop_smeared = core.invertPropagator(dirac, src, 1, 0)
    
    dst_seq = contract( "wtzyxijfc, ik -> wtzyxjkcf", prop_smeared.data.conj(), G5 )

    return dst_seq



#! GPT
U = g.convert(g.load("/home/jinchen/git/lat-software/LatCoding/conf/S8T8/wilson_b6.0"), g.double)

U_prime, U_trafo = g.gauge_fix(U, maxiter=1000, prec=1e-2)

U_trafo = g.identity(U_trafo)

# random source (spin-color field)
src = g.mspincolor(lat)
g.create.point(src, [1, 2, 1, 3])

p = {
    "mass": -0.038888,
    "csw_r": 1.02868,
    "csw_t": 1.02868,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}

w = g.qcd.fermion.wilson_clover(U, p)

# build solver using eo prec. and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-8, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator
prop = g.mspincolor(lat)
prop @= slv * src

gpt_dst_seq = create_bw_seq_gpt(prop, U_trafo, origin=[1,2,1,3])


#! Pyquda


mass = -0.038888
csw_r = 1.02868
csw_t = 1.02868
xi_0 = 1.0
nu = 1.0

latt_info = core.LatticeInfo([8, 8, 8, 8], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t)

gauge = io.readNERSCGauge("/home/jinchen/git/lat-software/LatCoding/conf/S8T8/wilson_b6.0")

dirac.loadGauge(gauge)


# same random source as GPT
point_prop = source.propagator(latt_info, "point", [1,2,1,3])

propag = core.invertPropagator(dirac, point_prop)

U_data = cp.zeros((8,8,8,8,3,3), dtype=cp.complex128)
U_data[..., 0,0] = 1
U_data[..., 1,1] = 1
U_data[..., 2,2] = 1
U_trafo = SimpleNamespace(data=U_data, latt_info=latt_info)

pyquda_dst_seq = create_bw_seq_pyquda(propag, U_trafo, origin=[1,2,1,3])

print(type(gpt_dst_seq))
print(type(pyquda_dst_seq))

diff = gpt_dst_seq.data.get() - pyquda_dst_seq.get()

print(max(abs(gpt_dst_seq.data.get()).flatten()))
print(max(abs(pyquda_dst_seq.get()).flatten()))

ratio = diff / gpt_dst_seq.data.get()

print(np.shape(ratio))
# print(ratio[0,0,0,0,0])
print(np.linalg.norm(ratio[0,0,0,0,0])**2)