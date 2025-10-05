# %%
import os
import numpy as np
from pyquda import init
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")

Fmunu_path = {
    (0, 1): [[X, Y, -X, -Y], [Y, -X, -Y, X], [-X, -Y, X, Y], [-Y, X, Y, -X]],
    (0, 2): [[X, Z, -X, -Z], [Z, -X, -Z, X], [-X, -Z, X, Z], [-Z, X, Z, -X]],
    (0, 3): [[X, T, -X, -T], [T, -X, -T, X], [-X, -T, X, T], [-T, X, T, -X]],
    (1, 2): [[Y, Z, -Y, -Z], [Z, -Y, -Z, Y], [-Y, -Z, Y, Z], [-Z, Y, Z, -Y]],
    (1, 3): [[Y, T, -Y, -T], [T, -Y, -T, Y], [-Y, -T, Y, T], [-T, Y, T, -Y]],
    (2, 3): [[Z, T, -Z, -T], [T, -Z, -T, Z], [-Z, -T, Z, T], [-T, Z, T, -Z]],
    (-1, -1): [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
}
Fmunu_coeff = [0.25, 0.25, 0.25, 0.25]

cfg = 0
gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")
clover_ij = gauge.loop(
    [
        Fmunu_path[(0, 1)],
        Fmunu_path[(0, 2)],
        Fmunu_path[(1, 2)],
        Fmunu_path[(-1, -1)],
    ],
    Fmunu_coeff,
)
Fij = (clover_ij[:3].data - clover_ij[:3].data.conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / 2j
clover_i4 = gauge.loop(
    [
        Fmunu_path[(0, 3)],
        Fmunu_path[(1, 3)],
        Fmunu_path[(2, 3)],
        Fmunu_path[(-1, -1)],
    ],
    Fmunu_coeff,
)
Fi4 = (clover_i4[:3].data - clover_i4[:3].data.conj().transpose(0, 1, 2, 3, 4, 5, 7, 6)) / 2j


# %%
print(np.shape(Fij))



# %%
