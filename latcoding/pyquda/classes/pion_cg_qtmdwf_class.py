import numpy as np

from pyquda_utils import core, gamma
from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.utils.tools import _get_xp_from_array, mpi_print, _asarray_on_queue
from latcoding.pyquda.utils.io_corr import save_c2pt_hdf5


my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#! Add PyQUDA gamma matrices by order
my_pyquda_gammas = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]
pyquda_gammas_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]
G5 = gamma.gamma(15)


def _gamma_matrix(gamma_like):
    if hasattr(gamma_like, "matrix"):
        return gamma_like.matrix
    return gamma_like


def _array_to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if type(arr).__module__.split(".")[0] == "cupy":
        return arr.get()
    # if type(arr).__module__.split(".")[0] == "dpnp":
    #     import dpnp
    #     return dpnp.asnumpy(arr)
    return np.asarray(arr)


def _gamma_on_backend(gamma_like, xp, ref_arr):
    return _asarray_on_queue(_gamma_matrix(gamma_like), xp, ref_arr)


"""
================================================================================
                                proton_TMD
================================================================================
"""
class pion_TMDWF_measurement():
    def __init__(self, parameters):

        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_pion(self, latt_info, prop_f, prop_b, phases, tag, src_mode="fixed", pion_interpolator=G5): 
        
        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        prop_b = boosted_smearing(prop_b, w=self.width, boost=self.neg_boost)
        mpi_print(latt_info, "Sink smearing completed")

        xp = _get_xp_from_array(prop_f.data)

        n_gamma = len(my_pyquda_gammas)
        G5_backend = _gamma_on_backend(G5, xp, prop_f.data)
        pyquda_gamma_ls = [_gamma_on_backend(gamma_pyq, xp, prop_f.data) for gamma_pyq in my_pyquda_gammas]

        # Source Dirac structure:
        # - fixed: use one fixed pion interpolator on source
        # - same_as_sink: use the same Gamma_g on source and sink
        # - dagger_of_sink: use gamma5 * Gamma_g^\dagger * gamma5 on source
        if src_mode == "fixed":
            pion_interpolator_backend = _gamma_on_backend(pion_interpolator, xp, prop_f.data)
            src_gamma_ls = [pion_interpolator_backend] * n_gamma
        elif src_mode == "same_as_sink":
            src_gamma_ls = pyquda_gamma_ls
        elif src_mode == "dagger_of_sink":
            src_gamma_ls = [
                xp.einsum("ab,bc,cd->ad", G5_backend, xp.swapaxes(gamma_g.conj(), 0, 1), G5_backend)
                for gamma_g in pyquda_gamma_ls
            ]
        else:
            raise ValueError(
                f"Invalid src_mode: {src_mode}. "
                "Expected one of ['fixed', 'same_as_sink', 'dagger_of_sink']."
            )

        phases = _asarray_on_queue(phases, xp, prop_f.data)
        bw_prop = xp.einsum("ij, wtzyxilab, kl -> wtzyxkjba", G5_backend, prop_b.data.conj(), G5_backend)

        corr_local = xp.zeros(
            (n_gamma, phases.shape[0], latt_info.global_size[3]),
            dtype=prop_f.data.dtype,
        )
        for gamma_idx, (sink_gamma, src_gamma) in enumerate(zip(pyquda_gamma_ls, src_gamma_ls)):
            bw_prop_g = xp.einsum("wtzyxjicf, im -> wtzyxjmcf", bw_prop, sink_gamma)
            temp_g = xp.einsum("wtzyxjiab, wtzyxilba, lj -> wtzyx", bw_prop_g, prop_f.data, src_gamma)
            corr_local[gamma_idx] = xp.einsum("qwtzyx, wtzyx -> qt", phases, temp_g)
            del bw_prop_g, temp_g

        corr = core.gatherLattice(_array_to_numpy(corr_local), [2, -1, -1, -1])
        
        if latt_info.mpi_rank == 0:
            save_c2pt_hdf5(corr, tag, my_gammas, self.plist)
        del corr, corr_local
    
        
    def create_TMD_Wilsonline_index_list_CG(self):
        index_list_trans0 = []
        index_list_trans1 = []
        
        for current_bz in range(0, self.b_z+1):
            for current_b_T in range(0, self.b_T+1):
                # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                index_list_trans0.append([current_b_T, current_bz, 0, 0])
                index_list_trans1.append([current_b_T, current_bz, 0, 1])

                if current_bz != 0:
                    index_list_trans0.append([current_b_T, -current_bz, 0, 0])
                    index_list_trans1.append([current_b_T, -current_bz, 0, 1])
                
        # Reorder index lists to minimize differences between adjacent indices
        def reorder_indices(index_list):
            # Sort by bT first, then bz to minimize jumps
            sorted_list = sorted(index_list, key=lambda x: (x[0], x[1]))
            reordered = []
            
            # Process pairs of indices to minimize differences
            i = 0
            while i < len(sorted_list)-1:
                curr = sorted_list[i]
                next = sorted_list[i+1]
                
                # If difference is more than 1 in either bT or bz, try to find better match
                if abs(curr[0] - next[0]) > 1 or abs(curr[1] - next[1]) > 1:
                    # Look ahead for better match
                    best_match = next
                    best_diff = max(abs(curr[0] - next[0]), abs(curr[1] - next[1]))
                    
                    for j in range(i+2, len(sorted_list)):
                        candidate = sorted_list[j]
                        diff = max(abs(curr[0] - candidate[0]), abs(curr[1] - candidate[1]))
                        if diff < best_diff:
                            best_match = candidate
                            best_diff = diff
                    
                    # Swap to get better ordering
                    if best_match != next:
                        idx = sorted_list.index(best_match)
                        sorted_list[i+1], sorted_list[idx] = sorted_list[idx], sorted_list[i+1]
                
                reordered.append(curr)
                i += 1
                
            if i < len(sorted_list):
                reordered.append(sorted_list[-1])
                
            return reordered
            
        index_list_trans0 = reorder_indices(index_list_trans0)
        index_list_trans1 = reorder_indices(index_list_trans1)
                
        return index_list_trans0, index_list_trans1
                    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_TMD_CG(self, prop_f, W_index, WL_indices_previous):
        current_b_T = W_index[0]
        current_bz = W_index[1]
        transverse_direction = W_index[3] # 0, 1
        Zdir = 2
        
        previous_b_T = WL_indices_previous[0]
        previous_bz = WL_indices_previous[1]
        
        prop_shift = prop_f.shift(round(current_b_T - previous_b_T), transverse_direction).shift(round(current_bz - previous_bz), Zdir)

        return prop_shift
    
    def create_PDF_Wilsonline_index_list(self):
        index_list = []
        
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
            index_list.append([0, current_bz, 0, 0])
            
        for current_bz in range(0, self.b_z + 1):
            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
            if current_bz != 0:
                index_list.append([0, -current_bz, 0, 0])
                    
        return index_list
    
    #! PyQUDA: create forward propagator for CG TMD, support +- shift
    def create_fw_prop_PDF_GI(self, gauge, prop_f_pyq, W_index, WL_indices_previous):

        current_bz = W_index[1]
        previous_bz = WL_indices_previous[1]

        #! PyQUDA: forward prop
        for spin in range(4):
            for color in range(3):
                fermion = prop_f_pyq.getFermion(spin, color)
                if current_bz - previous_bz == 0:
                    fermion_shift = fermion
                elif current_bz - previous_bz == 1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                elif current_bz - previous_bz == -1:
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 6) # -z direction
                else:
                    raise ValueError("Invalid shift for PDF Wilson line")
                #\psi'(x)=U_\mu(x)\psi(x+\hat\mu)0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
                prop_f_pyq.setFermion(fermion_shift, spin, color)

        return prop_f_pyq
