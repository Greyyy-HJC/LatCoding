import numpy as np

from pyquda.field import (
    LatticeInfo,
    LatticeGauge,
)
from pyquda_utils import source

def add_momentum_phase_to_gauge(latt_info: LatticeInfo, gauge: LatticeGauge, kvec, sign=-1):
    gauge_k = LatticeGauge(gauge.latt_info)
    gauge_k.data = gauge.data.copy()
    
    Gx, Gy, Gz, Gt = latt_info.global_size
    
    unit_converter = 2 * np.pi / np.array([Gx, Gy, Gz])

    ex = np.exp(1j * sign * kvec[0] * unit_converter[0])
    ey = np.exp(1j * sign * kvec[1] * unit_converter[1])
    ez = np.exp(1j * sign * kvec[2] * unit_converter[2])

    gauge_k.data[0] *= ex
    gauge_k.data[1] *= ey
    gauge_k.data[2] *= ez

    return gauge_k


def momentum_smearing_propagator(
    latt_info: LatticeInfo,
    gauge: LatticeGauge,
    kvec,          # [kx, ky, kz] in lattice units
    src_pos: list,
    rho: float,
    n_steps: int,
):
    # gauge with momentum phase
    gauge_k = add_momentum_phase_to_gauge(latt_info, gauge, kvec, sign=-1)

    point_prop = source.propagator(latt_info, "point", src_pos)
    smeared_prop = source.gaussianSmear(point_prop, gauge_k, rho, n_steps)
    return smeared_prop


def momentum_smearing_sink(
    latt_info: LatticeInfo,
    propagator,
    gauge: LatticeGauge,
    kvec,
    rho: float,
    n_steps: int,
):
    gauge_k = add_momentum_phase_to_gauge(latt_info, gauge, kvec, sign=-1)
    
    smeared_propagator = source.gaussianSmear(propagator, gauge_k, rho, n_steps)
    
    return smeared_propagator