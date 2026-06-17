'''
Modified by Jinchen He, 2025-11-21.
Fixed SYCL Queue Mismatch by ensuring all arrays share the propagator's queue.
Refactored by Gemini to use pyquda_utils.source.sequential12 for time slicing.
'''
import numpy as np

from pyquda.field import LatticePropagator
from pyquda_utils import core, gamma
from pyquda_utils.phase import MomentumPhase

from pyquda_utils.source import sequential12 
from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.utils.tools import _get_xp_from_array, _asarray_on_queue

# ---------- Precompute Constant Spin Matrices ----------
Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)
CgT5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7)
CgZ5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11)

Pp = (gamma.gamma(0) + gamma.gamma(8)) * 0.25
Szp = (gamma.gamma(0) - 1j*gamma.gamma(1) @ gamma.gamma(2))
Szm = (gamma.gamma(0) + 1j*gamma.gamma(1) @ gamma.gamma(2))
Sxp = (gamma.gamma(0) - 1j*gamma.gamma(2) @ gamma.gamma(4))
Sxm = (gamma.gamma(0) + 1j*gamma.gamma(2) @ gamma.gamma(4))
PpSzp = Pp @ Szp
PpSzm = Pp @ Szm
PpSxp = Pp @ Sxp
PpSxm = Pp @ Sxm

PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
    "PpUnpol": Pp,  
}

def _iter_bw_seq_raw(dirac, prop: LatticePropagator, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"):
    if interpolator == "5":
        gamma_insert = Cg5
    elif interpolator == "T5":
        gamma_insert = CgT5
    elif interpolator == "Z5":
        gamma_insert = CgZ5
    else:
        raise ValueError(f"Invalid interpolator: {interpolator}")

    xp = _get_xp_from_array(prop.data)
    latt_info = prop.latt_info
    GLt = latt_info.GLt

    prop = boosted_smearing(prop, w=sm_width, boost=sm_boost)

    for pol in pol_list:
        if flavor == 1:  # up quark insertion
            if latt_info.mpi_rank == 0:
                print(f"starting diquark contractions for up quark insertion and Polarization {pol}")
            src_seq = up_quark_insertion_pyquda(prop, prop, gamma_insert, PolProjections[pol])
        elif flavor == 2:  # down quark insertion
            if latt_info.mpi_rank == 0:
                print(f"starting diquark contractions for down quark insertion and Polarization {pol}")
            src_seq = down_quark_insertion_pyquda(prop, gamma_insert, PolProjections[pol])
        else:
            raise ValueError(f"Invalid flavor: {flavor}")

        t_source = origin[3]
        t_sink = (t_source + t_insert) % GLt
        src_seq_sliced = sequential12(src_seq, t_sink)
        seq_data = _asarray_on_queue(src_seq_sliced.data, xp, prop.data)

        mom_phase = MomentumPhase(latt_info).getPhase(momentum, x0=origin)
        mom_phase = _asarray_on_queue(mom_phase, xp, prop.data)

        G5 = _asarray_on_queue(gamma.gamma(15), xp, prop.data)
        data = xp.einsum("ij, wtzyx, wtzyxkjba -> wtzyxikab", G5, mom_phase, seq_data.conj())

        smearing_input = core.LatticePropagator(latt_info)
        smearing_input.data = data

        if latt_info.mpi_rank == 0:
            print(f"diquark contractions for Polarization {pol} done")

        src = boosted_smearing(smearing_input, w=sm_width, boost=sm_boost)
        yield core.invertPropagator(dirac, src, 1, 0)


def create_bw_seq_pyquda(dirac, prop: LatticePropagator, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"):
    """
    PyQUDA version: Build backward sequential source (Backend Agnostic).
    """
    dst_seq = []
    xp = None
    ref_data = None
    for raw in _iter_bw_seq_raw(
        dirac,
        prop,
        origin,
        sm_width,
        sm_boost,
        momentum,
        t_insert,
        pol_list,
        flavor,
        interpolator,
    ):
        if xp is None:
            xp = _get_xp_from_array(raw.data)
            ref_data = raw.data
        G5 = _asarray_on_queue(gamma.gamma(15), xp, raw.data)
        dst_seq.append(xp.einsum("wtzyxijfc, ik -> wtzyxjkcf", raw.data.conj(), G5))
    return _asarray_on_queue(dst_seq, xp, ref_data)


def create_bw_seq_raw_pyquda(dirac, prop: LatticePropagator, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"):
    """Build proton backward sequential propagators before final gamma5 conjugation."""
    return list(_iter_bw_seq_raw(
        dirac,
        prop,
        origin,
        sm_width,
        sm_boost,
        momentum,
        t_insert,
        pol_list,
        flavor,
        interpolator,
    ))


def create_meson_bw_seq_pyquda(
    dirac,
    prop: LatticePropagator,
    origin,
    momentum,
    t_insert,
    sink_gamma,
    sm_width=None,
    sm_boost=None,
):
    """
    Build and invert the meson fixed-sink backward sequential source.

    ``prop`` is expected to be the source-to-sink propagator with whatever sink
    smearing the caller wants to use for the sequential source.
    """
    xp = _get_xp_from_array(prop.data)
    latt_info = prop.latt_info
    t_sink = (origin[3] + t_insert) % latt_info.GLt

    src_seq_sliced = sequential12(prop, t_sink)
    seq_data = _asarray_on_queue(src_seq_sliced.data, xp, prop.data)

    mom_phase = MomentumPhase(latt_info).getPhases(
        [[momentum[0], momentum[1], momentum[2]]],
        origin,
    )[0]
    mom_phase = _asarray_on_queue(mom_phase, xp, prop.data)

    G5 = _asarray_on_queue(gamma.gamma(15), xp, prop.data)
    sink_gamma = _asarray_on_queue(sink_gamma, xp, prop.data)
    gamma_seq = xp.einsum("ab,cb,cd->ad", G5, sink_gamma.conj(), G5)

    seq_data = xp.einsum("wtzyx,wtzyxilab->wtzyxilab", mom_phase, seq_data)
    seq_data = xp.einsum("ij,wtzyxjlab->wtzyxilab", gamma_seq, seq_data)

    src_seq = core.LatticePropagator(latt_info)
    src_seq.data = seq_data
    if sm_width is not None and sm_boost is not None:
        src_seq = boosted_smearing(src_seq, w=sm_width, boost=sm_boost)
    return core.invertPropagator(dirac, src_seq, 1, 0)


def down_quark_insertion_pyquda(Q: LatticePropagator, Gamma, P):
    """
    PyQUDA version: Down quark insertion function.
    """
    # --- 1. Backend & Data Prep ---
    xp = _get_xp_from_array(Q.data)
    
    # Ensure Q is treated as an array on its native queue
    # We don't force move it, just wrap/ensure it's accessible
    q_data = Q.data 

    original_shape = q_data.shape 
    
    # Flatten: (Vol, spin_sink, spin_src, color_sink, color_src)
    flat_Q = q_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare Gamma and P matrices on the SAME QUEUE ---
    def to_backend_matrix(g):
        val = gamma.gamma(g) if isinstance(g, int) else g
        # CRITICAL FIX: Use the queue from q_data
        return _asarray_on_queue(val, xp, q_data)

    G_mat = to_backend_matrix(Gamma)
    P_mat = to_backend_matrix(P)
    Gt_mat = G_mat.T 

    # --- 3. Precompute spin space matrix operations ---
    PDu = xp.einsum('ij, ...jiab -> ...ab', P_mat, flat_Q)
    GtDG = xp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, flat_Q, G_mat)
    GtD = xp.einsum('ij, ...jkab -> ...ikab', Gt_mat, flat_Q)
    PDG = xp.einsum('ij, ...jkab, kl -> ...ilab', P_mat, flat_Q, G_mat)

    # --- 4. Color tensor contraction ---
    # CRITICAL FIX: Create eps on Host first, then move to specific Queue
    eps_host = np.zeros((3, 3, 3), dtype=q_data.dtype)
    eps_host[0, 1, 2] = eps_host[1, 2, 0] = eps_host[2, 0, 1] = 1
    eps_host[2, 1, 0] = eps_host[1, 0, 2] = eps_host[0, 2, 1] = -1
    
    # Move eps to the correct queue
    eps = _asarray_on_queue(eps_host, xp, q_data)

    # Term 1
    term1 = xp.einsum('abc, def, ...fc, ...uveb -> ...uvad', eps, eps, PDu, GtDG)

    # Term 2
    term2 = xp.einsum('abc, def, ...ujec, ...jkfb -> ...ukad', eps, eps, GtD, PDG)

    # Combine
    D_flat = term2 - term1

    # --- 5. Post-processing ---
    D_transposed = xp.swapaxes(D_flat, -4, -3)
    final_data = D_transposed.reshape(original_shape)

    # --- 6. Package ---
    R = core.LatticePropagator(Q.latt_info)
    R.data = final_data
    return R


def up_quark_insertion_pyquda(Qu: LatticePropagator, Qd: LatticePropagator, Gamma, P):
    """
    PyQUDA version: Up quark insertion function.
    """
    # --- 1. Backend & Data Prep ---
    xp = _get_xp_from_array(Qu.data)
    
    qu_data = Qu.data
    qd_data = Qd.data # Assume Qd is on same queue as Qu. Usually true.
    
    original_shape = qu_data.shape
    Qu_flat = qu_data.reshape(-1, 4, 4, 3, 3)
    Qd_flat = qd_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare matrices on the SAME QUEUE ---
    def to_backend_matrix(g):
        val = gamma.gamma(g) if isinstance(g, int) else g
        # CRITICAL FIX: Use the queue from qu_data
        return _asarray_on_queue(val, xp, qu_data)

    G_mat = to_backend_matrix(Gamma)
    P_mat = to_backend_matrix(P)
    Gt_mat = G_mat.T

    # --- 3. Precompute intermediate terms ---
    GtDG = xp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, Qd_flat, G_mat)
    PDu = xp.einsum('ij, ...jkab -> ...ikab', P_mat, Qu_flat)
    DuP = xp.einsum('...jkab, kl -> ...jlab', Qu_flat, P_mat)
    TrDuP = xp.einsum('...kjab, jk -> ...ab', Qu_flat, P_mat)

    # --- 4. Epsilon contraction ---
    # CRITICAL FIX: Create eps on Host, then move to Queue
    eps_host = np.zeros((3, 3, 3), dtype=qu_data.dtype)
    eps_host[0, 1, 2] = eps_host[1, 2, 0] = eps_host[2, 0, 1] = 1
    eps_host[2, 1, 0] = eps_host[1, 0, 2] = eps_host[0, 2, 1] = -1
    
    # Move eps to the correct queue
    eps = _asarray_on_queue(eps_host, xp, qu_data)

    # Term 1
    T1_scalar = xp.einsum('...mnbe, ...mnad -> ...bead', GtDG, Qu_flat)
    R1_pre = xp.einsum('abc, def, ...bead -> ...cf', eps, eps, T1_scalar)
    R1 = xp.einsum('ij, ...cf -> ...ijcf', P_mat, R1_pre)

    # Term 2
    R2 = xp.einsum('abc, def, ...ad, ...jibe -> ...ijcf', eps, eps, TrDuP, GtDG)

    # Term 3
    R3 = xp.einsum('abc, def, ...ikad, ...jkbe -> ...ijcf', eps, eps, PDu, GtDG)

    # Term 4
    R4 = xp.einsum('abc, def, ...kiad, ...klbe -> ...ilcf', eps, eps, GtDG, DuP)

    # Total Sum
    D_total = -1 * (R1 + R2 + R3 + R4)

    # --- 5. Post-processing ---
    D_final = xp.swapaxes(D_total, -1, -2)
    final_data = D_final.reshape(original_shape)

    # --- 6. Return result ---
    R = core.LatticePropagator(Qu.latt_info)
    R.data = final_data
    
    return R
