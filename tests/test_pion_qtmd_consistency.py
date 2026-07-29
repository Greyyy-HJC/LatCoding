from pathlib import Path

import h5py
import numpy as np

from latcoding.pyquda.utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PHYSICAL_FROM_PYQUDA,
    PYQUDA_GAMMA_IDS,
)
from latcoding.pyquda.utils.io_corr import (
    save_connected_qtmd_hdf5,
    save_qTMDWF_hdf5_noRoll,
)
from latcoding.pyquda.utils.tools import append_sample_log_entry, read_sample_log_entries


ROOT = Path(__file__).resolve().parents[1]


def test_sample_log_uses_exact_lines_and_durable_deduplication(tmp_path):
    log = tmp_path / "nested" / "samples.log"
    assert read_sample_log_entries(log) == set()
    assert append_sample_log_entry(log, "base1")
    assert append_sample_log_entry(log, "base10")
    assert not append_sample_log_entry(log, "base1")
    assert read_sample_log_entries(log) == {"base1", "base10"}


def test_canonical_gamma_basis_metadata():
    assert len(GAMMA_LABELS) == len(PYQUDA_GAMMA_IDS) == 16
    assert GAMMA_LABELS[0] == "5"
    assert PYQUDA_GAMMA_IDS[0] == 15
    assert PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("Y5"), GAMMA_LABELS.index("Y5")] == -1
    assert PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("T5"), GAMMA_LABELS.index("T5")] == -1


def test_connected_qtmd_dense_schema(tmp_path):
    tag = tmp_path / "pion"
    momentum = [[0, 0, 0, 0], [0, 0, 1, 0]]
    wilson = [[0, 0, 0, 0], [0, 1, 0, 0]]
    corr = np.arange(2 * 2 * 16 * 6).reshape(2, 2, 16, 6)
    save_connected_qtmd_hdf5(
        corr,
        str(tag),
        momentum,
        wilson,
        4,
        attrs={"active_boost": [0, 0, -3]},
    )

    with h5py.File(f"{tag}.h5", "r") as h5_file:
        assert h5_file.attrs["qtmd_hdf5_schema"] == "connected_qtmd_dense_v1"
        assert h5_file.attrs["corr_axes"] == "wilson,momentum,gamma,time"
        np.testing.assert_array_equal(h5_file["corr"], corr)
        np.testing.assert_array_equal(h5_file["momentum_list"], momentum)
        np.testing.assert_array_equal(h5_file["wilson_index_list"], wilson)
        assert [value.decode() for value in h5_file["gamma_list"][:]] == list(GAMMA_LABELS)


def test_qtmdwf_one_file_contains_all_gamma_groups(tmp_path):
    tag = tmp_path / "wavefunction"
    momentum = [[0, 0, 0, 0]]
    wilson = [[0, 0, 0, 0]]
    corr = np.arange(16 * 5).reshape(1, 1, 16, 5)
    save_qTMDWF_hdf5_noRoll(corr, str(tag), list(GAMMA_LABELS), momentum, wilson)

    with h5py.File(f"{tag}.h5", "r") as h5_file:
        assert h5_file.attrs["qtmdwf_hdf5_schema"] == "qtmdwf_hierarchical_v1"
        assert set(h5_file["SP"]) == set(GAMMA_LABELS)
        for gamma_idx, gamma_label in enumerate(GAMMA_LABELS):
            dataset = h5_file[f"SP/{gamma_label}/PX0PY0PZ0/b_X/eta0/bT0/bz0"]
            np.testing.assert_array_equal(dataset, corr[0, 0, gamma_idx])


def test_pion_runner_routes_spectator_and_active_boost_lines():
    source = (ROOT / "examples/pyquda/measurement/pion_cg_qtmd.py").read_text()
    sequential = source[source.index("# Fixed-sink sequential propagator"):source.index("qext_xyz")]
    contraction = source[source.index("pion_TMDs = measurement.contract_qTMD_CG"):]
    assert "prop_pos.copy()" in sequential
    assert 'boost=parameters["pos_boost"]' in sequential
    assert 'parameters["neg_boost"]' in sequential
    assert "prop_neg," in contraction
    assert "prop_sink_smeared" not in sequential


def test_boosted_smearing_batches_full_propagator():
    source = (ROOT / "latcoding/pyquda/utils/boosted_smearing.py").read_text()
    assert "len(src.field_shape)" in source
    assert "_boosted_smearing_field(src" in source
    assert "for s in range(Ns)" not in source
    assert "if float(w) == 0.0" not in source
