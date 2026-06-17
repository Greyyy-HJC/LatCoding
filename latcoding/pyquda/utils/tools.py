#!/usr/bin/env python3

import os

# ---------- Backend Helpers (consistent with boosted_smearing_pyquda) ----------
def _get_xp_from_array(a):
    """Return the base module of the array's type, e.g. cupy / numpy / dpnp / torch."""
    # Handle case where data might be wrapped or None, default to numpy if unsure
    if a is None:
        return __import__("numpy")
    base = type(a).__module__.split('.')[0]
    return __import__(base)

def _ensure_backend(x, xp):
    """Move x to the same backend as xp if needed."""
    # Check if x is already on the correct backend
    if type(x).__module__.split('.')[0] == xp.__name__:
        return x
    # Handle transfer
    if hasattr(xp, "asarray"):
        return xp.asarray(x)
    if xp.__name__ == "torch":
        return xp.as_tensor(x)
    return xp.array(x)

# --- HELPER: Ensure new arrays are on the EXACT SAME QUEUE as the data ---
def _asarray_on_queue(val, xp, ref_arr):
    """
    Creates an array 'val' on the same backend and SYCL queue as 'ref_arr'.
    """
    # 1. If usage is dpnp, strictly enforce the sycl_queue
    if xp.__name__ == 'dpnp' and hasattr(ref_arr, 'sycl_queue'):
        return xp.asarray(val, sycl_queue=ref_arr.sycl_queue)
    
    # 2. Fallback for standard numpy/cupy or if ref_arr has no queue info
    return xp.asarray(val)


def gamma_matrix_to_backend(gamma_like, xp, ref_arr=None, dtype=None):
    """Convert a PyQUDA gamma object or matrix to an array on the requested backend."""
    if hasattr(gamma_like, "matrix"):
        gamma_like = gamma_like.matrix
    if ref_arr is not None:
        gamma_arr = _asarray_on_queue(gamma_like, xp, ref_arr)
    else:
        gamma_arr = xp.asarray(gamma_like)
    if dtype is not None:
        gamma_arr = gamma_arr.astype(dtype, copy=False)
    return gamma_arr

def mpi_print(latt_info, message):
    if latt_info.mpi_rank == 0:
        print(message)


def timing_enabled():
    value = os.environ.get("PYQUDA_MEASUREMENT_TIMERS", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def mpi_timer_print(latt_info, label, seconds, **fields):
    if not timing_enabled() or latt_info.mpi_rank != 0:
        return
    items = [f"TIMER {label}", f"seconds={seconds:.6f}"]
    for key, value in fields.items():
        if isinstance(value, float):
            value = f"{value:.6f}"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        items.append(f"{key}={value}")
    print(" ".join(items), flush=True)


def srcLoc_distri_eq(L, src_origin):
    source_positions = []
    i_src = 0
    div = 4
    for i in range(div):
        for j in range(div):
            for k in range(div):
                for l in range(div):
                    source_positions += [[round(i*L[0]/div+src_origin[0])%L[0], round(j*L[1]/div+src_origin[1])%L[1], round(k*L[2]/div+src_origin[2])%L[2], round(l*L[3]/div+src_origin[3])%L[3]]]
    return source_positions
