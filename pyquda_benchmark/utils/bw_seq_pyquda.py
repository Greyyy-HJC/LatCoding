import cupy as cp

from pyquda_utils import core, gamma
from boosted_smearing_pyquda import boosted_smearing
from pyquda_utils.phase import MomentumPhase

Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)

Pp = (gamma.gamma(0) + gamma.gamma(8)) * 0.25
Szp = (gamma.gamma(0) - 1j*gamma.gamma(1) @ gamma.gamma(2))
Szm = (gamma.gamma(0) + 1j*gamma.gamma(1) @ gamma.gamma(2))
Sxp = (gamma.gamma(0) - 1j*gamma.gamma(2) @ gamma.gamma(4))
Sxm = (gamma.gamma(0) + 1j*gamma.gamma(2) @ gamma.gamma(4))
PpSzp = Pp @ Szp
PpSzm = Pp @ Szm
PpSxp = Pp @ Sxp
PpSxm = Pp @ Sxm


def create_bw_seq_pyquda(prop, trafo, origin, sm_width, sm_boost, momentum, t_insert):
    """
    PyQUDA version: Build backward sequential source.
    
    Parameters:
        prop: LatticePropagator
            Input forward propagator (u quark).
        origin: list/tuple
            Source coordinates in format [x, y, z, t] (GPT style).
        t_insert: int
            Insertion time offset relative to source time (t_sink = (t_source + t_insert) % Lt).
        momentum: list/tuple
            Momentum in format [px, py, pz, pt] (i.e., x, y, z, t).
        trafo: transformation parameter
            Transformation parameter for boosted smearing.
        sm_width: float
            Smearing width parameter.
        sm_boost: float
            Boost parameter for smearing.
            
    Returns:
        smearing_input: LatticePropagator
            Sequential source for inversion.
    """
    
    latt_info = prop.latt_info
    Lt = latt_info.Lt
    
    prop = boosted_smearing(trafo, prop, w=sm_width, boost=sm_boost)

    # --- 1. Perform baryon contraction (Up Quark Insertion) ---
    src_seq = up_quark_insertion_pyquda(prop, prop, Cg5, PpSzp)
    
    # --- 2. Time slicing ---
    # origin format [x, y, z, t] -> time is at index 3
    t_source = origin[3] 
    t_sink = (t_source + t_insert) % Lt
    
    # Get data copy (GPU)
    seq_data = src_seq.data.copy()
    
    # Zero out non-insertion time slices
    times = cp.arange(Lt)
    mask = (times != t_sink) 
    seq_data[:, mask] = 0    
    
    # --- 3. Create momentum phase ---
    # momentum: [px, py, pz, pt] (x, y, z, t)
    # PyQUDA MomentumPhase typically expects order [t, z, y, x]
    # We map momentum: x->mom[3], y->mom[2], z->mom[1], t->mom[0]
    
    # Generate phase
    mom_phase = MomentumPhase(latt_info).getPhase(momentum)
    
    G5 = gamma.gamma(15)
    
    smearing_input = cp.einsum("jk,wtzyx,wtzyxkiba->wtzyxjiba", G5, mom_phase, seq_data)

    return smearing_input

def down_quark_insertion_pyquda(Q, Gamma, P):
    """
    PyQUDA version: Down quark insertion function (Down Quark Insertion / Baryon Block Contraction).
    
    Formula:
    R = Transpose_spin( - epsilon_abc * epsilon_def * ( 
        Trace_spin(P*Q)_{fc} * (Gt * Q * G)_{eb} - (Gt * Q)_{ec} * (P * Q * G)_{fb} 
    ))
    where Gt = Gamma^T.
    
    Parameters:
        Q: pyquda.core.LatticePropagator
            Input quark propagator.
        Gamma: int or array-like
            Inserted Gamma matrix (integer index 0-15 or 4x4 matrix).
        P: int or array-like
            Projection/polarization matrix (integer index 0-15 or 4x4 matrix).
            
    Returns:
        R: pyquda.core.LatticePropagator
            Contracted diquark propagator.
    """
    
    # --- 1. Prepare data and dimension handling ---
    # Q.data shape is typically (2, Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)
    # Reshape to (Vol, Ns, Ns, Nc, Nc) for batch matrix operations
    q_data = Q.data
    if not isinstance(q_data, cp.ndarray):
        q_data = cp.array(q_data) # Ensure on GPU

    # Save original shape for later restoration (w, t, z, y, x, ...)
    original_shape = q_data.shape 
    
    # Flatten all spacetime dimensions to first dimension
    # Shape: (Vol, spin_sink, spin_src, color_sink, color_src)
    flat_Q = q_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare Gamma and P matrices ---
    def to_cupy_matrix(g):
        if isinstance(g, int):
            # Use pyquda built-in library to get Gamma matrix
            return cp.array(gamma.gamma(g))
        return cp.asarray(g)

    G_mat = to_cupy_matrix(Gamma)
    P_mat = to_cupy_matrix(P)
    Gt_mat = G_mat.T  # Matrix transpose

    # --- 3. Precompute spin space matrix operations ---
    # Use einsum for broadcasting, preserving color and batch dimensions
    # flat_Q indices: ...jkab -> ... (batch), j(sink spin), k(src spin), a(sink color), b(src color)
    
    # PDu: Trace_spin(P * Q) -> Result is Color Matrix (Vol, 3, 3)
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

    # --- 4. Color tensor contraction ---
    # Construct Epsilon tensor
    eps = cp.zeros((3, 3, 3), dtype=q_data.dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # Target formula (combined with GPT logic):
    # term = - eps_abc * eps_def * ( PDu_fc * GtDG_eb - GtD_ec * PDG_fb )
    # Note: sign1*sign2 in GPT code corresponds to eps*eps.
    # GPT accumulation logic is -sign1*sign2, so there's an overall -1 factor.
    # We compute both terms separately, then subtract (Term2 - Term1) to handle the negative sign.

    # Mapping indices for eps_abc(sink) and eps_def(src):
    # sink colors: a, b, c. src colors: d, e, f.
    
    # Term 1: eps_abc * eps_def * PDu_{fc} * GtDG_{eb}
    # PDu indices: f(sink), c(src) -> Actually PDu comes from Q, Q indices are (sink, src).
    # Carefully check GPT indices: PDu[i2[2], i1[2]] -> PDu[f, c] (if i2=def, i1=abc)
    # GtDG[i2[1], i1[1]] -> GtDG[e, b]
    # Result indices: a(sink), d(src). Spin indices come from GtDG (u, v).
    # Einsum path: contract b, c, e, f. Keep a, d, u, v.
    term1 = cp.einsum('abc, def, ...fc, ...uveb -> ...uvad', eps, eps, PDu, GtDG)

    # Term 2: eps_abc * eps_def * (GtD * PDG)
    # Note: this includes spin matrix multiplication: GtD(..., u, j, ...) * PDG(..., j, v, ...)
    # GPT: GtD[i2[1], i1[2]] * PDG[i2[2], i1[1]]
    # color map: GtD -> sink e, src c. PDG -> sink f, src b.
    # Spin map: GtD(u, j) * PDG(j, v) -> (u, v)
    term2 = cp.einsum('abc, def, ...ujec, ...jkfb -> ...ukad', eps, eps, GtD, PDG)

    # Combine results: Result = Term2 - Term1
    # (corresponds to GPT's -1 * (Term1 - Term2))
    D_flat = term2 - term1

    # --- 5. Post-processing ---
    
    # Spin Transpose (GPT's g.transpose typically swaps Spin source/sink)
    # Current D_flat shape: (Vol, s_sink, s_src, c_sink, c_src)
    # Swap spin dimensions (axis -4 and -3)
    D_transposed = cp.swapaxes(D_flat, -4, -3)

    # Restore spacetime dimensions
    # Shape becomes (w, t, z, y, x, 4, 4, 3, 3)
    final_data = D_transposed.reshape(original_shape)

    # --- 6. Package and return ---
    R = core.LatticePropagator(Q.latt_info)
    R.data = final_data
    
    return R


def up_quark_insertion_pyquda(Qu, Qd, Gamma, P):
    """
    PyQUDA version: Up quark insertion function (Up Quark Insertion).
    
    Parameters:
        Qu: pyquda.core.LatticePropagator
            Input up quark propagator.
        Qd: pyquda.core.LatticePropagator
            Input down quark propagator.
        Gamma: int or array-like
            Inserted Gamma matrix (integer index 0-15 or 4x4 matrix).
        P: int or array-like
            Projection/polarization matrix (integer index 0-15 or 4x4 matrix).
            
    Returns:
        R: pyquda.core.LatticePropagator
            Contracted result after up quark insertion.
    """
    # --- 1. Prepare data ---
    # Ensure data is on GPU
    qu_data = cp.array(Qu.data) if not isinstance(Qu.data, cp.ndarray) else Qu.data
    qd_data = cp.array(Qd.data) if not isinstance(Qd.data, cp.ndarray) else Qd.data
    
    original_shape = qu_data.shape
    # Flatten: (Batch, Ns, Ns, Nc, Nc)
    # Indices: ...jkab (j=sink spin, k=src spin, a=sink color, b=src color)
    Qu_flat = qu_data.reshape(-1, 4, 4, 3, 3)
    Qd_flat = qd_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare matrices ---
    def to_cupy_matrix(g):
        if isinstance(g, int):
            return cp.array(gamma.gamma(g))
        return cp.asarray(g)

    G_mat = to_cupy_matrix(Gamma)
    P_mat = to_cupy_matrix(P)
    Gt_mat = G_mat.T

    # --- 3. Precompute intermediate terms ---
    
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

    # --- 4. Epsilon contraction (Main Contraction) ---
    eps = cp.zeros((3, 3, 3), dtype=qu_data.dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # Term 1: P * spin_trace(GtDG[b, e] * Du[a, d].T)
    # GPT logic: Trace(A * B^T) = Sum_{i,j} A_{ij} * B_{ij} (element-wise product sum)
    # PyQUDA original incorrect logic: A_{mn} * B_{nm} (this is Trace(A*B))
    # Correction: GtDG indices ...mnbe, Qu indices ...mnad (keep mn order consistent)
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

    # --- 5. Post-processing (Indices Adjustment) ---
    # Current indices: (..., sink_spin, src_spin, sink_color, src_color)
    # Swap color indices to match GPT output: sink, src -> src, sink
    D_final = cp.swapaxes(D_total, -1, -2)

    # Restore shape
    final_data = D_final.reshape(original_shape)

    # --- 6. Return result ---
    R = core.LatticePropagator(Qu.latt_info)
    R.data = final_data
    
    return R