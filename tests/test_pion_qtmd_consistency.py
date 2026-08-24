from pathlib import Path

import h5py
import numpy as np

from latcoding.pyquda.utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    PHYSICAL_FROM_PYQUDA,
    PYQUDA_GAMMA_IDS,
)
from latcoding.pyquda.utils.io_corr import (
    save_qTMD_pion_hdf5_noRoll,
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


def test_connected_qtmd_grouped_schema(tmp_path):
    tag = tmp_path / "pion"
    momentum = [[0, 0, 0, 0], [0, 0, 1, 0]]
    wilson = [[0, 0, 0, 0], [0, 1, 0, 1]]
    tsep = 4
    corr = np.arange(2 * 2 * 16 * (tsep + 1), dtype=np.complex128).reshape(2, 2, 16, tsep + 1)
    save_qTMD_pion_hdf5_noRoll(
        corr,
        str(tag),
        list(GAMMA_LABELS),
        momentum,
        wilson,
        tsep,
        attrs={"source_interpolator": "T5", "sink_interpolator": "T5", "active_boost": [0, 0, -3]},
    )

    with h5py.File(f"{tag}.h5", "r") as h5_file:
        assert h5_file.attrs["qtmd_hdf5_schema"] == "pion_qtmd_groups_v3"
        path = "T5/T5/PX0PY0PZ1/tsep4/eta0/bT0/bz1/bTdirY/T"
        np.testing.assert_array_equal(h5_file[path], corr[1, 1, list(GAMMA_LABELS).index("T")])
        assert [value.decode() for value in h5_file["_meta/gamma_list"][:]] == list(GAMMA_LABELS)


def test_qtmdwf_grouped_schema(tmp_path):
    tag = tmp_path / "wavefunction"
    momentum = [[0, 0, 1, 0]]
    wilson = [[0, 0, 0, 0], [1, -2, 0, 1]]
    corr = np.arange(2 * 1 * 16 * 5, dtype=np.complex128).reshape(2, 1, 16, 5)
    save_qTMDWF_hdf5_noRoll(
        corr,
        str(tag),
        list(GAMMA_LABELS),
        momentum,
        wilson,
        attrs={"source_interpolator": "T5"},
    )

    with h5py.File(f"{tag}.h5", "r") as h5_file:
        assert h5_file.attrs["qtmdwf_hdf5_schema"] == "pion_qtmdwf_groups_v3"
        assert h5_file.attrs["source_role"] == "local interpolator"
        assert h5_file.attrs["sink_role"] == "nonlocal operator"
        path = "T5/T_nonlocal/PX0PY0PZ1/eta0/bT1/bz-2/bTdirY"
        np.testing.assert_array_equal(
            h5_file[path], corr[1, 0, list(GAMMA_LABELS).index("T")]
        )
        assert [value.decode() for value in h5_file["_meta/gamma_list"][:]] == list(GAMMA_LABELS)


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
