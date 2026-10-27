"""One circular correlation for every CG qTMDWF displacement.

The coordinate-gauge line is a pure shift, so

    C(b) = sum_x Tr[Gamma_src gamma5 G(x+b)^dagger gamma5 Gamma_sink G(x)]

is a correlation along the transverse axis and z. Index ``(bz, bT)`` matches
``LatticePropagator.shift(+bT, transverse).shift(+bz, z)`` followed by
``contract_pion_2pt``.
"""

import numpy as np
from pyquda import getMPIComm
from pyquda_utils import core

from latcoding.pyquda.utils.pion_utils import (
    G5,
    _array_to_numpy,
    _gamma_on_backend,
    gamma_stack,
    source_gamma_stack,
)
from latcoding.pyquda.utils.tools import _get_xp_from_array


# Lattice mu = x,y,z,t maps onto lexicographic axes t,z,y,x.
_LEXICO_AXIS_OF_MU = (3, 2, 1, 0)
# Lexicographic axis t,z,y,x maps onto grid indices x,y,z,t.
_GRID_INDEX_OF_LEXICO_AXIS = (3, 2, 1, 0)


def rank_keeps_gathered_plane(coord, grid, gather_dims):
    """Only one rank per gathered subvolume may enter the final sum."""
    replicated = [dim for dim in gather_dims if int(grid[dim]) > 1]
    if not replicated:
        return True
    return all(int(coord[dim]) == 0 for dim in replicated)


def _gather_color_key(coord, grid, gather_dims):
    color = 0
    color_stride = 1
    key = 0
    key_stride = 1
    for dim in range(4):
        if dim in gather_dims:
            key += int(coord[dim]) * key_stride
            key_stride *= int(grid[dim])
        else:
            color += int(coord[dim]) * color_stride
            color_stride *= int(grid[dim])
    return color, key


def _stitch_lexico(blocks, coords, grid, gather_dims):
    local = blocks[0].shape
    out_size = []
    for axis in range(4):
        grid_index = _GRID_INDEX_OF_LEXICO_AXIS[axis]
        factor = grid[grid_index] if grid_index in gather_dims else 1
        out_size.append(local[axis] * factor)
    out = np.empty((*out_size, *local[4:]), dtype=blocks[0].dtype)
    for block, coord in zip(blocks, coords):
        slices = []
        for axis in range(4):
            grid_index = _GRID_INDEX_OF_LEXICO_AXIS[axis]
            if grid_index in gather_dims:
                start = coord[grid_index] * local[axis]
                slices.append(slice(start, start + local[axis]))
            else:
                slices.append(slice(None))
        out[tuple(slices)] = block
    return out


def _shift_axes(transverse_mu):
    transverse_axis = _LEXICO_AXIS_OF_MU[int(transverse_mu) % 4]
    z_axis = 1
    if transverse_axis == z_axis:
        return (z_axis,)
    return (z_axis, transverse_axis)


def _gather_lexico_axes(field, axes):
    """Gather shifted axes onto one rank of each unaffected-coordinate group."""
    from pyquda_comm import getGridCoord, getGridSize

    grid = tuple(int(value) for value in getGridSize())
    coord = tuple(int(value) for value in getGridCoord())
    gather_dims = {_GRID_INDEX_OF_LEXICO_AXIS[axis] for axis in axes}
    if all(grid[dim] == 1 for dim in gather_dims):
        return field
    host = np.ascontiguousarray(_array_to_numpy(field))
    color, key = _gather_color_key(coord, grid, gather_dims)
    comm = getMPIComm()
    sub = comm.Split(int(color), int(key))
    try:
        blocks = sub.gather(host, root=0)
        coords = sub.gather(coord, root=0)
    finally:
        sub.Free()
    if key != 0:
        return None
    stitched = _stitch_lexico(blocks, coords, grid, gather_dims)
    return _get_xp_from_array(field).asarray(stitched)


def qtmdwf_correlation_plane(backward, forward, gamma5, sink_gammas, source_gamma, transverse_mu):
    """Correlate lexicographic propagators.

    Arrays have shape ``(t,z,y,x,spin,spin,color,color)`` and are complete
    along z and the transverse axis. The result is ``(t, z, transverse, gamma)``
    when those axes differ, with ``FullField.shift(+bT).shift(+bz)`` stored at
    ``[:, bz % Lz, bT % Ltrans]``.
    """
    xp = _get_xp_from_array(backward)
    axes = _shift_axes(transverse_mu)
    backward_line = xp.einsum(
        "ij,tzyxmlca,kl->tzyxkjca", gamma5, backward.conj(), gamma5, optimize=True
    )
    forward_aligned = xp.swapaxes(forward, -1, -2)
    correlated = None
    for color_a in range(3):
        for color_b in range(3):
            left = backward_line[..., color_a, color_b]
            right = forward_aligned[..., color_a, color_b]
            spectrum = xp.fft.fftn(left, axes=axes)[..., :, :, None, None] * xp.conj(
                xp.fft.fftn(xp.conj(right), axes=axes)
            )[..., None, None, :, :]
            term = xp.fft.ifftn(spectrum, axes=axes)
            correlated = term if correlated is None else correlated + term
    drop = tuple(axis for axis in range(1, 4) if axis not in axes)
    reduced = correlated.sum(axis=drop)
    if source_gamma.ndim == 2:
        corr = xp.einsum("t...jpil,gpi,lj->t...g", reduced, sink_gammas, source_gamma, optimize=True)
    else:
        corr = xp.einsum("t...jpil,gpi,glj->t...g", reduced, sink_gammas, source_gamma, optimize=True)
    spatial = [axis for axis in range(1, 4) if axis in axes]
    return corr, spatial


def _sample_plane(plane, spatial, transverse_mu, bT, bz):
    transverse_axis = _LEXICO_AXIS_OF_MU[int(transverse_mu) % 4]
    z_axis = 1
    index = [slice(None)]
    for axis in spatial:
        length = plane.shape[len(index)]
        shift = 0
        if axis == z_axis:
            shift += int(bz)
        if axis == transverse_axis:
            shift += int(bT)
        index.append(shift % length)
    return plane[tuple(index)]


def contract_cg_qtmdwf_direction(
    latt_info, prop_forward, prop_backward, phases, src_gamma, wilson_indices
):
    """Contract one transverse direction in the existing Wilson-index order.

    Rank 0 receives ``(n_wilson, n_momentum, n_gamma, Lt)``. Other ranks
    receive ``None``.
    """
    if not np.allclose(_array_to_numpy(phases), 1):
        raise NotImplementedError(
            "qTMDWF correlation supports the zero-momentum phase used by quarkCG"
        )
    xp = _get_xp_from_array(prop_forward.data)
    gamma5 = _gamma_on_backend(G5, xp, prop_forward.data)
    sink_gammas = gamma_stack(prop_forward.data)
    source_gammas = source_gamma_stack(src_gamma, sink_gammas, prop_forward.data)
    source_gamma = source_gammas[0] if xp.all(source_gammas == source_gammas[:1]) else source_gammas

    forward = prop_forward.lexico(False)
    backward = prop_backward.lexico(False)
    transverse_mu = int(wilson_indices[0][3])
    axes = _shift_axes(transverse_mu)
    n_gamma = int(sink_gammas.shape[0])
    t_local = int(forward.shape[0])
    n_wilson = len(wilson_indices)
    local = xp.zeros((n_wilson, n_gamma, 1, t_local), dtype=forward.dtype)

    from pyquda_comm import getGridCoord, getGridSize

    grid = tuple(int(value) for value in getGridSize())
    coord = tuple(int(value) for value in getGridCoord())
    gather_dims = {_GRID_INDEX_OF_LEXICO_AXIS[axis] for axis in axes}
    if rank_keeps_gathered_plane(coord, grid, gather_dims):
        gathered_backward = _gather_lexico_axes(backward, axes)
        gathered_forward = _gather_lexico_axes(forward, axes)
        if gathered_backward is not None and gathered_forward is not None:
            plane, spatial = qtmdwf_correlation_plane(
                gathered_backward,
                gathered_forward,
                gamma5,
                sink_gammas,
                source_gamma,
                transverse_mu,
            )
            for i_wilson, wilson_index in enumerate(wilson_indices):
                sample = _sample_plane(
                    plane, spatial, transverse_mu, wilson_index[0], wilson_index[1]
                )
                local[i_wilson, :, 0] = xp.swapaxes(sample, 0, 1)
    gathered = core.gatherLattice(_array_to_numpy(local), [3, -1, -1, -1])
    if latt_info.mpi_rank != 0:
        return None
    return np.transpose(gathered, (0, 2, 1, 3))
