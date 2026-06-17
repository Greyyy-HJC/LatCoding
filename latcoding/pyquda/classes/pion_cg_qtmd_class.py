"""
Pion connected qTMD and PDF contractions in PyQUDA.

This module is the pion analogue of ``proton_qTMD_pyquda.py``.  The main
structural difference is that a pion correlator contains one quark line and one
antiquark line, while the proton code contracts three quark lines.  For this
reason the pion code uses ``pos_boost`` and ``neg_boost`` to describe the
smearing boosts of the quark and antiquark lines, matching the convention used
in ``pion_qTMDWF_pyquda.py``.

Conventions and propagators
---------------------------
Let S_q(x, y) be the quark propagator from source y to sink x.  The antiquark
line is constructed with gamma5 hermiticity,

    S_anti(x, y) = gamma5 * S_q(x, y)^dagger * gamma5.

In the code this is implemented by ``meson_backward_line``.  The PyQUDA
propagator layout is kept in even-odd lattice order,

    prop.data[w, t, z, y, x_cb, spin_sink, spin_src, color_sink, color_src].

The helper ``gamma_stack`` prepares the 16 bilinear gamma matrices in the same
order as the existing proton and pion-TMDWF workflows:

    5, T, T5, X, X5, Y, Y5, Z, Z5, I, SXT, SXY, SXZ, SYT, SYZ, SZT.

Two-point function
------------------
For a source gamma Gamma_src and each sink gamma Gamma_g, the connected pion
two-point function is

    C2_g(p, t) =
        sum_x exp(-i p . (x - x0))
        Tr_spin,color[
            S_anti(x, x0) Gamma_g S_q(x, x0) Gamma_src
        ].

The default ``src_gamma='fixed_g5'`` gives the usual pseudoscalar pion source,
while all 16 sink gamma structures are still scanned and saved.  This is cheap
and useful for diagnostics or later operator studies.

Three-point function and fixed-sink sequential source
-----------------------------------------------------
The connected pion qTMD three-point function starts from a source at x0, an
operator insertion at x = (tau, x), and a fixed sink at y = (tsep, y):

    C3_g(q, b, tau; pf, tsep) =
        sum_x sum_y
        exp(-i q . (x - x0)) exp(-i pf . (y - x0))
        Tr[
            S_anti(y, x0) Gamma_sink
            S_q(y, x) Gamma_g O_b S_q(x, x0) Gamma_src
        ].

Gamma_sink is the fixed pion sink interpolator used to build the sequential
source.  Gamma_g is the scanned insertion gamma.  O_b is the nonlocal qTMD/PDF
operator applied to the forward quark line at the insertion.

The sink sum is absorbed into a fixed-sink sequential propagator.  In
``create_meson_bw_seq_pyquda`` the right-hand side is built on the sink time
slice,

    eta_seq(y; pf, tsep) =
        delta_{t_y,tsep}
        phase_pf(y - x0)
        Gamma_seq S_neg(y, x0),

    Gamma_seq = gamma5 Gamma_sink^dagger gamma5,

where ``phase_pf`` is produced by ``MomentumPhase`` with the same sign
convention used by the application.  The sequential propagator solves

    D S_seq = eta_seq.

After the inversion this module forms the antiquark-like backward line

    S_seq_anti(x, x0; pf, tsep) = gamma5 S_seq(x, x0)^dagger gamma5.

The contraction used in the code is therefore the sink-summed form

    C3_g(q, b, tau; pf, tsep) =
        sum_x exp(-i q . (x - x0))
        Tr[
            S_seq_anti(x, x0; pf, tsep)
            Gamma_g
            O_b S_q(x, x0)
            Gamma_src
        ].

CG qTMD operator
----------------
For the CG qTMD path, O_b is currently a coordinate-gauge style displacement
without explicit gauge links:

    O_b S_q(x, x0) = S_q(x + bT * e_perp + bz * ez, x0).

The transverse direction is scanned over x and y, stored as ``b_X`` and
``b_Y`` in the HDF5 output.  The code contracts both directions separately so
successive shifts stay small.

PDF operators
-------------
The PDF path is the straight-z special case with bT = 0:

    b = bz * ez.

The CG_PDF path uses ordinary lattice shifts without explicit gauge links.

Output shape convention
-----------------------
The contraction routines naturally return arrays as

    [Wilson_index, gamma, momentum, time].

The Perlmutter application transposes them before calling the shared qTMD HDF5
writer, which expects

    [Wilson_index, momentum, gamma, time].

This explicit transpose is important: without it, a single gamma file can be
written with the wrong momentum/gamma interpretation.
"""

import numpy as np

from pyquda_utils import core
from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.utils.tools import _asarray_on_queue, _get_xp_from_array, mpi_print
from latcoding.pyquda.utils.pion_utils import (
    contract_pion_2pt,
    gamma_stack,
    meson_backward_line,
    my_gammas,
    source_gamma_stack,
)


def save_c2pt_hdf5(*args, **kwargs):
    raise NotImplementedError(
        "This draft example needs an HDF5 writer that has not been migrated into latcoding yet."
    )


class pion_TMD:
    def __init__(self, parameters):
        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]

        self.pf = parameters["pf"]
        self.qlist = parameters["qext"]
        self.qlist_PDF = parameters.get("qext_PDF", self.qlist)
        self.pilist = parameters["p_2pt"]

        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]

        self.t_insert = parameters["t_insert"]
        self.save_propagators = parameters["save_propagators"]

    def contract_2pt_pion(self, latt_info, prop_f, prop_b, phases, tag, src_gamma="fixed_g5"):
        mpi_print(latt_info, "Begin pion sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        prop_b = boosted_smearing(prop_b, w=self.width, boost=self.neg_boost)
        mpi_print(latt_info, "Pion sink smearing completed")

        corr = contract_pion_2pt(latt_info, prop_f, prop_b, phases, src_gamma=src_gamma)

        if latt_info.mpi_rank == 0:
            save_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr

    def contract_qTMD_CG(self, latt_info, prop_f, seq_bw_prop, phases, W_index_list_dir0, W_index_list_dir1, src_gamma="fixed_g5"):
        xp = _get_xp_from_array(prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        sink_gamma_ls = gamma_stack(prop_f.data)
        source_gamma_ls = source_gamma_stack(src_gamma, sink_gamma_ls, prop_f.data)
        seq_bw_line = meson_backward_line(seq_bw_prop)

        pion_TMDs = []
        W_index_list = W_index_list_dir0 + W_index_list_dir1

        tmd_forward_prop_dir0 = prop_f.copy()
        for iW, W_index in enumerate(W_index_list_dir0):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir0[iW - 1]
            tmd_forward_prop_dir0 = self.create_fw_prop_TMD_CG(tmd_forward_prop_dir0, W_index, W_index_previous)
            pion_TMDs.append(self._contract_qTMD_one_shift(seq_bw_line, tmd_forward_prop_dir0, sink_gamma_ls, source_gamma_ls, phases))
        del tmd_forward_prop_dir0

        tmd_forward_prop_dir1 = prop_f.copy()
        for iW, W_index in enumerate(W_index_list_dir1):
            mpi_print(latt_info, f"Contract pion qTMD CG {iW + 1 + len(W_index_list_dir0)}/{len(W_index_list)} {W_index}")
            W_index_previous = [0, 0, 0, 0] if iW == 0 else W_index_list_dir1[iW - 1]
            tmd_forward_prop_dir1 = self.create_fw_prop_TMD_CG(tmd_forward_prop_dir1, W_index, W_index_previous)
            pion_TMDs.append(self._contract_qTMD_one_shift(seq_bw_line, tmd_forward_prop_dir1, sink_gamma_ls, source_gamma_ls, phases))
        del tmd_forward_prop_dir1

        return np.array(pion_TMDs)

    def contract_PDF(self, latt_info, prop_f, seq_bw_prop, phases, W_index_list, src_gamma="fixed_g5"):
        xp = _get_xp_from_array(prop_f.data)
        phases = _asarray_on_queue(phases, xp, prop_f.data)
        sink_gamma_ls = gamma_stack(prop_f.data)
        source_gamma_ls = source_gamma_stack(src_gamma, sink_gamma_ls, prop_f.data)
        seq_bw_line = meson_backward_line(seq_bw_prop)

        pion_PDFs = []
        pdf_forward_prop = prop_f.copy()
        for iW, W_index in enumerate(W_index_list):
            mpi_print(latt_info, f"Contract pion PDF CG {iW + 1}/{len(W_index_list)} {W_index}")
            if W_index[1] == 0:
                W_index_previous = [0, 0, 0, 0]
                pdf_forward_prop = prop_f.copy()
            elif W_index[1] > 0:
                W_index_previous = W_index_list[iW - 1]
            elif W_index[1] == -1:
                W_index_previous = [0, 0, 0, 0]
                pdf_forward_prop = prop_f.copy()
            else:
                W_index_previous = W_index_list[iW - 1]

            pdf_forward_prop = self.create_fw_prop_TMD_CG(pdf_forward_prop, W_index, W_index_previous)
            pion_PDFs.append(self._contract_qTMD_one_shift(seq_bw_line, pdf_forward_prop, sink_gamma_ls, source_gamma_ls, phases))
        del pdf_forward_prop

        return np.array(pion_PDFs)

    def _contract_qTMD_one_shift(self, seq_bw_line, shifted_prop, sink_gamma_ls, source_gamma_ls, phases):
        xp = _get_xp_from_array(shifted_prop.data)
        sink_inserted = xp.einsum("wtzyxjicf,gim->gwtzyxjmcf", seq_bw_line, sink_gamma_ls, optimize=True)
        corr_local = xp.einsum(
            "gwtzyxjiab,wtzyxilba,glj->gwtzyx",
            sink_inserted,
            shifted_prop.data,
            source_gamma_ls,
            optimize=True,
        )
        corr = xp.einsum("qwtzyx,gwtzyx->gqt", phases, corr_local, optimize=True)
        return core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])

    def create_TMD_Wilsonline_index_list_CG(self):
        index_list_trans0 = []
        index_list_trans1 = []

        for current_bz in range(0, self.b_z + 1):
            for current_b_T in range(0, self.b_T + 1):
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])

        return self._reorder_wilson_indices(index_list_trans0), self._reorder_wilson_indices(index_list_trans1)

    def _reorder_wilson_indices(self, index_list):
        sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
        reordered = []
        i = 0
        while i < len(sorted_list) - 1:
            current = sorted_list[i]
            next_index = sorted_list[i + 1]
            if abs(current[0] - next_index[0]) > 1 or abs(current[1] - next_index[1]) > 1:
                best_match = next_index
                best_diff = max(abs(current[0] - next_index[0]), abs(current[1] - next_index[1]))
                for candidate in sorted_list[i + 2 :]:
                    diff = max(abs(current[0] - candidate[0]), abs(current[1] - candidate[1]))
                    if diff < best_diff:
                        best_match = candidate
                        best_diff = diff
                if best_match != next_index:
                    best_index = sorted_list.index(best_match)
                    sorted_list[i + 1], sorted_list[best_index] = sorted_list[best_index], sorted_list[i + 1]
            reordered.append(current)
            i += 1

        if i < len(sorted_list):
            reordered.append(sorted_list[-1])
        return reordered

    def create_fw_prop_TMD_CG(self, prop_f, W_index, W_index_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3]
        z_direction = 2

        previous_b_T = W_index_previous[0]
        previous_bz = W_index_previous[1]

        return prop_f.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), z_direction)

    def create_PDF_Wilsonline_index_list(self):
        index_list = []

        for current_bz in range(0, self.b_z + 1):
            index_list.append([0, current_bz, 0, 0])

        for current_bz in range(0, self.b_z + 1):
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])

        return index_list
