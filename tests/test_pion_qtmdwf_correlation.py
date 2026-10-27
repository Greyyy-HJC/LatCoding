import numpy as np
from pyquda_utils import gamma

from latcoding.pyquda.utils.fermion_bilinear_basis import GAMMA_LABELS, PYQUDA_GAMMA_IDS
from latcoding.pyquda.utils.pion_qtmdwf_correlation import (
    qtmdwf_correlation_plane,
    rank_keeps_gathered_plane,
)


def _matrix(gamma_id):
    value = gamma.gamma(gamma_id)
    return np.asarray(value.matrix if hasattr(value, "matrix") else value)


def _reference(backward, forward, gamma5, sink, source, bT, bz, transverse_axis):
    shifted = np.roll(np.roll(backward, -bT, axis=transverse_axis), -bz, axis=1)
    backward_line = np.einsum("ij,tzyxmlca,kl->tzyxkjca", gamma5, shifted.conj(), gamma5)
    inserted = np.einsum("tzyxjicf,im->tzyxjmcf", backward_line, sink)
    site = np.einsum("tzyxjiab,tzyxilba,lj->tzyx", inserted, forward, source)
    return site.sum(axis=(1, 2, 3))


def test_correlation_plane_matches_shifted_trace_for_every_gamma():
    rng = np.random.default_rng(0)
    shape = (2, 4, 3, 4, 4, 4, 3, 3)
    backward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    forward = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    gamma5 = _matrix(15)
    sink_gammas = np.stack([_matrix(gamma_id) for gamma_id in PYQUDA_GAMMA_IDS])
    source = _matrix(PYQUDA_GAMMA_IDS[GAMMA_LABELS.index("T5")])

    for transverse_mu, transverse_axis in ((0, 3), (1, 2)):
        plane, spatial = qtmdwf_correlation_plane(
            backward, forward, gamma5, sink_gammas, source, transverse_mu
        )
        plane = np.asarray(plane)
        for bT in (0, 1, 3):
            for bz in (0, 1, -1, 2):
                for gamma_index, sink in enumerate(sink_gammas):
                    expected = _reference(
                        backward, forward, gamma5, sink, source, bT, bz, transverse_axis
                    )
                    index = [slice(None)]
                    for axis in spatial:
                        length = plane.shape[len(index)]
                        shift = 0
                        if axis == 1:
                            shift += bz
                        if axis == transverse_axis:
                            shift += bT
                        index.append(shift % length)
                    index.append(gamma_index)
                    got = plane[tuple(index)]
                    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_gathered_plane_is_not_multiplied_by_the_spatial_grid():
    grid = (2, 1, 2, 1)
    gather_dims = {0, 2}
    owners = [
        rank_keeps_gathered_plane((gx, 0, gz, 0), grid, gather_dims)
        for gx in range(2)
        for gz in range(2)
    ]
    assert owners.count(True) == 1

    split_y = (2, 2, 2, 1)
    owners = [
        rank_keeps_gathered_plane((gx, gy, gz, 0), split_y, gather_dims)
        for gx in range(2)
        for gy in range(2)
        for gz in range(2)
    ]
    assert owners.count(True) == split_y[1]
