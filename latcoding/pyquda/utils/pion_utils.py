import numpy as np

from pyquda_utils import core, gamma
from latcoding.pyquda.utils.tools import _asarray_on_queue, _get_xp_from_array


my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
my_pyquda_gammas = [
    gamma.gamma(15),
    gamma.gamma(8),
    gamma.gamma(7),
    gamma.gamma(1),
    gamma.gamma(14),
    gamma.gamma(2),
    gamma.gamma(13),
    gamma.gamma(4),
    gamma.gamma(11),
    gamma.gamma(0),
    gamma.gamma(9),
    gamma.gamma(3),
    gamma.gamma(5),
    gamma.gamma(10),
    gamma.gamma(6),
    gamma.gamma(12),
]
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


def gamma_stack(reference_array):
    xp = _get_xp_from_array(reference_array)
    return xp.stack([_gamma_on_backend(gamma_matrix, xp, reference_array) for gamma_matrix in my_pyquda_gammas])


def gamma_from_label(label):
    if label not in my_gammas:
        raise ValueError(f"Invalid gamma label: {label}. Expected one of {my_gammas}.")
    return my_pyquda_gammas[my_gammas.index(label)]


def source_gamma_stack(src_gamma, sink_gamma_ls, reference_array):
    xp = _get_xp_from_array(reference_array)
    gamma5 = _gamma_on_backend(G5, xp, reference_array)

    if src_gamma == "fixed_g5":
        source_gamma_ls = sink_gamma_ls.copy()
        source_gamma_ls[:] = gamma5
        return source_gamma_ls
    if src_gamma == "same_as_sink":
        return sink_gamma_ls.copy()
    if src_gamma == "dagger_of_sink":
        return xp.einsum("ab,gbc,cd->gad", gamma5, xp.swapaxes(sink_gamma_ls.conj(), 1, 2), gamma5, optimize=True)
    if src_gamma in my_gammas:
        source_gamma = _gamma_on_backend(gamma_from_label(src_gamma), xp, reference_array)
        source_gamma_ls = sink_gamma_ls.copy()
        source_gamma_ls[:] = source_gamma
        return source_gamma_ls

    raise ValueError(
        f"Invalid src_gamma: {src_gamma}. "
        "Use a gamma label or one of ['fixed_g5', 'same_as_sink', 'dagger_of_sink']."
    )


def meson_backward_line(prop):
    xp = _get_xp_from_array(prop.data)
    gamma5 = _gamma_on_backend(G5, xp, prop.data)
    return xp.einsum("ij,wtzyxilab,kl->wtzyxkjba", gamma5, prop.data.conj(), gamma5, optimize=True)


def contract_pion_2pt(latt_info, prop_forward, prop_backward, phases, src_gamma="fixed_g5"):
    return contract_pion_2pt_multi_src_gamma(
        latt_info,
        prop_forward,
        prop_backward,
        phases,
        [src_gamma],
    )[src_gamma]


def contract_pion_2pt_multi_src_gamma(latt_info, prop_forward, prop_backward, phases, src_gammas):
    xp = _get_xp_from_array(prop_forward.data)
    sink_gamma_ls = gamma_stack(prop_forward.data)
    source_gamma_ls_by_src = {
        src_gamma: source_gamma_stack(src_gamma, sink_gamma_ls, prop_forward.data)
        for src_gamma in src_gammas
    }
    phases = _asarray_on_queue(phases, xp, prop_forward.data)

    backward_line = meson_backward_line(prop_backward)
    corr_local_by_src = {
        src_gamma: xp.zeros(
            (len(sink_gamma_ls), phases.shape[0], latt_info.size[3]),
            dtype=prop_forward.data.dtype,
        )
        for src_gamma in src_gammas
    }

    for gamma_idx, sink_gamma in enumerate(sink_gamma_ls):
        sink_inserted = xp.einsum("wtzyxjicf,im->wtzyxjmcf", backward_line, sink_gamma, optimize=True)
        for src_gamma in src_gammas:
            corr_site = xp.einsum(
                "wtzyxjiab,wtzyxilba,lj->wtzyx",
                sink_inserted,
                prop_forward.data,
                source_gamma_ls_by_src[src_gamma][gamma_idx],
                optimize=True,
            )
            corr_local_by_src[src_gamma][gamma_idx] = xp.einsum("qwtzyx,wtzyx->qt", phases, corr_site, optimize=True)
            del corr_site
        del sink_inserted

    corr_by_src = {
        src_gamma: core.gatherLattice(_array_to_numpy(corr_local), [2, -1, -1, -1])
        for src_gamma, corr_local in corr_local_by_src.items()
    }
    del corr_local_by_src, source_gamma_ls_by_src, backward_line
    return corr_by_src
