"""Lightweight metadata for the historical 16-element PyQUDA gamma basis."""

import numpy as np

GAMMA_BASIS_SCHEMA = "pyquda_bitmask16_with_physics_transform_v1"
GAMMA_LABELS = (
    "5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I",
    "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT",
)
PYQUDA_GAMMA_IDS = (
    15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12,
)
PHYSICAL_FROM_PYQUDA = np.eye(16, dtype=np.complex128)
PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("Y5"), GAMMA_LABELS.index("Y5")] = -1
PHYSICAL_FROM_PYQUDA[GAMMA_LABELS.index("T5"), GAMMA_LABELS.index("T5")] = -1


def basis_metadata():
    """Return HDF5-safe datasets describing raw and physical gamma ordering."""
    return {
        "gamma_list": np.asarray(GAMMA_LABELS, dtype="S"),
        "gamma_pyquda_ids": np.asarray(PYQUDA_GAMMA_IDS, dtype=np.int32),
        "physical_gamma_list": np.asarray(GAMMA_LABELS, dtype="S"),
        "physical_from_pyquda": PHYSICAL_FROM_PYQUDA.copy(),
    }


def basis_attrs():
    """Return stable scalar attributes for gamma convention provenance."""
    return {
        "gamma_basis_schema": GAMMA_BASIS_SCHEMA,
        "gamma_basis_order": ",".join(GAMMA_LABELS),
        "physical_transform_definition": (
            "Gamma_physical[A]=sum_B physical_from_pyquda[A,B]*Gamma_raw[B]"
        ),
        "gamma5_definition": "gamma1*gamma2*gamma3*gamma4",
        "axial_definition": "gamma_mu*gamma5",
        "raw_tensor_definition": "0.5*[gamma_mu,gamma_nu]",
        "hermitian_tensor_from_raw_factor": "1j",
    }
