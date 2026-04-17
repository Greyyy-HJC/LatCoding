
import cupy as cp
from pyquda_utils import core, gamma
import gpt as g

Cg5_pyquda = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)

Pp = (gamma.gamma(0) + gamma.gamma(8)) * 0.25
Szp = (gamma.gamma(0) - 1j*gamma.gamma(1) @ gamma.gamma(2))
Szm = (gamma.gamma(0) + 1j*gamma.gamma(1) @ gamma.gamma(2))
Sxp = (gamma.gamma(0) - 1j*gamma.gamma(2) @ gamma.gamma(4))
Sxm = (gamma.gamma(0) + 1j*gamma.gamma(2) @ gamma.gamma(4))
PpSzp_pyquda = Pp @ Szp


Cg5_gpt = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp_gpt = Pp * Szp


def down_quark_insertion_pyquda(Q, Gamma, P):
    """
    PyQUDA 版下夸克插入函数 (Down Quark Insertion / Baryon Block Contraction)。
    
    计算公式对应: 
    R = Transpose_spin( - epsilon_abc * epsilon_def * ( 
        Trace_spin(P*Q)_{fc} * (Gt * Q * G)_{eb} - (Gt * Q)_{ec} * (P * Q * G)_{fb} 
    ))
    其中 Gt = Gamma^T.
    
    参数:
        Q: pyquda.core.LatticePropagator
            输入的夸克传播子。
        Gamma: int 或 array-like
            插入的 Gamma 矩阵 (整数索引 0-15 或 4x4 矩阵)。
        P: int 或 array-like
            投影矩阵/极化矩阵 (整数索引 0-15 或 4x4 矩阵)。
            
    返回:
        R: pyquda.core.LatticePropagator
            缩并后的双夸克传播子 (Diquark propagator)。
    """
    
    # --- 1. 准备数据与维度处理 ---
    # Q.data 形状通常为 (2, Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)
    # 我们将其 reshape 为 (Vol, Ns, Ns, Nc, Nc) 以便进行批量矩阵运算
    q_data = Q.data
    if not isinstance(q_data, cp.ndarray):
        q_data = cp.array(q_data) # 确保在 GPU 上

    # 保存原始形状以便最后还原 (w, t, z, y, x, ...)
    original_shape = q_data.shape 
    vol_shape = original_shape[:-4] # (w, t, z, y, x)
    
    # 展平所有时空维度到第一维
    # Shape: (Vol, spin_sink, spin_src, color_sink, color_src)
    flat_Q = q_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. 准备 Gamma 和 P 矩阵 ---
    def to_cupy_matrix(g):
        if isinstance(g, int):
            # 使用 pyquda 内置库获取 Gamma 矩阵
            return cp.array(gamma.gamma(g))
        return cp.asarray(g)

    G_mat = to_cupy_matrix(Gamma)
    P_mat = to_cupy_matrix(P)
    Gt_mat = G_mat.T  # 矩阵转置

    # --- 3. 预计算自旋空间矩阵乘法 (Spin Matrix Operations) ---
    # 使用 einsum 进行广播乘法，保留 color 维度和 batch 维度
    # flat_Q indices: ...jkab -> ... (batch), j(sink spin), k(src spin), a(sink color), b(src color)
    
    # PDu: Trace_spin(P * Q) -> 结果是 Color Matrix (Vol, 3, 3)
    # P_ij * Q_ji -> Sum over i,j
    # P(u,v) * Q(...v,u, a, b) -> ...ab
    PDu = cp.einsum('ij, ...jiab -> ...ab', P_mat, flat_Q)

    # GtDG: G.T * Q * G -> Propagator (Vol, 4, 4, 3, 3)
    # Gt(i,j) * Q(..., j, k, ...) * G(k, l) -> result(..., i, l, ...)
    GtDG = cp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, flat_Q, G_mat)

    # GtD: G.T * Q -> Propagator
    GtD = cp.einsum('ij, ...jkab -> ...ikab', Gt_mat, flat_Q)

    # PDG: P * Q * G -> Propagator
    PDG = cp.einsum('ij, ...jkab, kl -> ...ilab', P_mat, flat_Q, G_mat)

    # --- 4. 颜色张量缩并 (Color Contraction) ---
    # 构造 Epsilon 张量
    eps = cp.zeros((3, 3, 3), dtype=q_data.dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # 目标公式 (结合 gpt 逻辑):
    # term = - eps_abc * eps_def * ( PDu_fc * GtDG_eb - GtD_ec * PDG_fb )
    # 注意: gpt 代码中的 sign1*sign2 对应 eps*eps。
    # gpt 累加逻辑为 -sign1*sign2，所以整体有个 -1 因子。
    # 我们分别计算两项，然后相减 (Term2 - Term1) 来处理这个负号。

    # Mapping indices for eps_abc(sink) and eps_def(src):
    # sink colors: a, b, c. src colors: d, e, f.
    
    # Term 1: eps_abc * eps_def * PDu_{fc} * GtDG_{eb}
    # PDu indices: f(sink), c(src) -> 实际上 PDu 来自 Q，Q indices是 (sink, src)。
    # 仔细核对 gpt 索引: PDu[i2[2], i1[2]] -> PDu[f, c] (如果 i2=def, i1=abc)
    # GtDG[i2[1], i1[1]] -> GtDG[e, b]
    # 结果索引: a(sink), d(src). Spin indices 来自 GtDG (u, v).
    # Einsum path: contract b, c, e, f. Keep a, d, u, v.
    term1 = cp.einsum('abc, def, ...fc, ...uveb -> ...uvad', eps, eps, PDu, GtDG)

    # Term 2: eps_abc * eps_def * (GtD * PDG)
    # 注意这里包含自旋矩阵乘法: GtD(..., u, j, ...) * PDG(..., j, v, ...)
    # gpt: GtD[i2[1], i1[2]] * PDG[i2[2], i1[1]]
    # color map: GtD -> sink e, src c. PDG -> sink f, src b.
    # Spin map: GtD(u, j) * PDG(j, v) -> (u, v)
    term2 = cp.einsum('abc, def, ...ujec, ...jkfb -> ...ukad', eps, eps, GtD, PDG)

    # 组合结果: Result = Term2 - Term1
    # (对应 gpt 的 -1 * (Term1 - Term2))
    D_flat = term2 - term1

    # --- 5. 后处理 ---
    
    # Spin Transpose (gpt 的 g.transpose 通常交换 Spin source/sink)
    # 当前 D_flat 形状: (Vol, s_sink, s_src, c_sink, c_src)
    # 交换 spin 维度 (axis -4 和 -3)
    D_transposed = cp.swapaxes(D_flat, -4, -3)

    # 还原时空维度
    # 形状变回 (w, t, z, y, x, 4, 4, 3, 3)
    final_data = D_transposed.reshape(original_shape)

    # --- 6. 封装返回 ---
    R = core.LatticePropagator(Q.latt_info)
    R.data = final_data
    
    return R


def up_quark_insertion_pyquda(Qu, Qd, Gamma, P):
    """
    PyQUDA 版上夸克插入函数 (Up Quark Insertion)。
    """
    # --- 1. 准备数据 ---
    # 确保数据在 GPU 上
    qu_data = cp.array(Qu.data) if not isinstance(Qu.data, cp.ndarray) else Qu.data
    qd_data = cp.array(Qd.data) if not isinstance(Qd.data, cp.ndarray) else Qd.data
    
    original_shape = qu_data.shape
    # 展平: (Batch, Ns, Ns, Nc, Nc)
    # Indices: ...jkab (j=sink spin, k=src spin, a=sink color, b=src color)
    Qu_flat = qu_data.reshape(-1, 4, 4, 3, 3)
    Qd_flat = qd_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. 准备矩阵 ---
    def to_cupy_matrix(g):
        if isinstance(g, int):
            return cp.array(gamma.gamma(g))
        return cp.asarray(g)

    G_mat = to_cupy_matrix(Gamma)
    P_mat = to_cupy_matrix(P)
    Gt_mat = G_mat.T

    # --- 3. 预计算中间项 ---
    
    # GtDG = G.T * Qd * G
    # ...jkab -> ...ilab
    GtDG = cp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, Qd_flat, G_mat)

    # PDu = P * Qu
    PDu = cp.einsum('ij, ...jkab -> ...ikab', P_mat, Qu_flat)

    # DuP = Qu * P
    DuP = cp.einsum('...jkab, kl -> ...jlab', Qu_flat, P_mat)

    # TrDuP = Trace_spin(Qu * P) -> Color Matrix (Batch, 3, 3)
    # Trace(Qu * P) = Sum_k (Qu * P)_kk = Sum_k Sum_j Qu_kj P_jk
    # ...kjab, jk -> ...ab
    TrDuP = cp.einsum('...kjab, jk -> ...ab', Qu_flat, P_mat)

    # --- 4. Epsilon 缩并 (Main Contraction) ---
    eps = cp.zeros((3, 3, 3), dtype=qu_data.dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # Term 1: P * spin_trace(GtDG[b, e] * Du[a, d].T)
    # GPT 逻辑: Trace(A * B^T) = Sum_{i,j} A_{ij} * B_{ij} (逐元素乘积之和)
    # PyQUDA 原错误逻辑: A_{mn} * B_{nm} (这是 Trace(A*B))
    # 修正: GtDG indices ...mnbe, Qu indices ...mnad (保持 mn 顺序一致)
    T1_scalar = cp.einsum('...mnbe, ...mnad -> ...bead', GtDG, Qu_flat)
    
    # Contract with epsilons
    R1_pre = cp.einsum('abc, def, ...bead -> ...cf', eps, eps, T1_scalar)
    R1 = cp.einsum('ij, ...cf -> ...ijcf', P_mat, R1_pre)

    # Term 2: Transpose( TrDuP[a, d] * GtDG[b, e] )
    # TrDuP[a, d] is scalar. GtDG[b, e] is matrix. Result is scalar * GtDG.T
    # ...jibe (ji implies transpose of spin)
    R2 = cp.einsum('abc, def, ...ad, ...jibe -> ...ijcf', eps, eps, TrDuP, GtDG)

    # Term 3: PDu[a, d] * GtDG[b, e].T
    # PDu ...ikad, GtDG ...jkbe (using jk matches transpose row selection if result is ij)
    # ...ik * ...jk -> ...ij
    R3 = cp.einsum('abc, def, ...ikad, ...jkbe -> ...ijcf', eps, eps, PDu, GtDG)

    # Term 4: GtDG[a, d].T * DuP[b, e]
    # GtDG ...kiad (ki implies input for transpose), DuP ...klbe
    # ...ki * ...kl -> ...il (matches GtDG.T * DuP)
    R4 = cp.einsum('abc, def, ...kiad, ...klbe -> ...ilcf', eps, eps, GtDG, DuP)

    # Total Sum
    # Coefficient is -sign1 * sign2.
    D_total = -1 * (R1 + R2 + R3 + R4)

    # --- 5. 后处理 (Indices Adjustment) ---
    # Current indices: (..., sink_spin, src_spin, sink_color, src_color)
    # Swap color indices to match GPT output: sink, src -> src, sink
    D_final = cp.swapaxes(D_total, -1, -2)

    # Restore shape
    final_data = D_final.reshape(original_shape)

    # --- 6. 返回结果 ---
    R = core.LatticePropagator(Qu.latt_info)
    R.data = final_data
    
    return R


def down_quark_insertion_gpt(Q, Gamma, P):
    #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
    # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
    
    eps = g.epsilon(Q.otype.shape[2])
    
    R = g.lattice(Q)
    
    PDu = g(g.spin_trace(P*Q))

    GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

    GtDG = g.separate_color(GtDG)
    PDu = g.separate_color(PDu)
    
    GtD = g.eval(g.transpose(Gamma)*Q)
    PDG = g.eval(P*Q*Gamma)
    
    GtD = g.separate_color(GtD)
    PDG = g.separate_color(PDG)
    
    D = {x: g.lattice(GtDG[x]) for x in GtDG}

    for d in D:
        D[d][:] = 0
        
    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i1[0], i2[0]] += -sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
            
    g.merge_color(R, D)
    return R

#Qlua definition, reproduce the results as Chroma difinition
def up_quark_insertion_gpt(Qu, Qd, Gamma, P):

    eps = g.epsilon(Qu.otype.shape[2])
    R = g.lattice(Qu)

    Du_sep = g.separate_color(Qu)
    GDd = g.eval(Gamma * Qd)
    GDd = g.separate_color(GDd)

    PDu = g.eval(P*Qu)
    PDu = g.separate_color(PDu)

    # ut
    DuP = g.eval(Qu * P)
    DuP = g.separate_color(DuP)
    TrDuP = g(g.spin_trace(Qu * P))
    TrDuP = g.separate_color(TrDuP)
    
    # s2ds1b
    GtDG = g.eval(g.transpose(Gamma)*Qd*Gamma)
    GtDG = g.separate_color(GtDG)

    #sum color indices
    D = {x: g.lattice(GDd[x]) for x in GDd}
    for d in D:
        D[d][:] = 0

    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i2[2], i1[2]] += -sign1 * sign2 * (P * g.spin_trace(GtDG[i1[1],i2[1]]*g.transpose(Du_sep[i1[0],i2[0]]))
                                + g.transpose(TrDuP[i1[0],i2[0]] * GtDG[i1[1],i2[1]])
                                + PDu[i1[0],i2[0]] * g.transpose(GtDG[i1[1],i2[1]])
                                + g.transpose(GtDG[i1[0],i2[0]]) * DuP[i1[1],i2[1]])
    
    g.merge_color(R, D)

    return R
