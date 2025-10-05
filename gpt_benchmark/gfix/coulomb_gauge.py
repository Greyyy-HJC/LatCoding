"""
added by Jinchen on 2023.12.14

The only difference between Coulomb gauge and Landau gauge is the functional form, here we just sum over the spatial components, i.e. [:-1] in the return of __call__ function.
"""

import gpt as g
import numpy as np
from gpt.core.group import differentiable_functional


class coulomb(differentiable_functional):
    def __init__(self, U):
        self.U = U

    def __call__(self, V):
        V = g.util.from_list(V)
        functional = 0.0
        for mu in range(3):  # Only spatial directions
            transformed_links = g.qcd.gauge.transformed(self.U, V)[mu]
            functional += g.sum(g.trace(transformed_links))
        return functional.real * (-2.0)

    @differentiable_functional.single_field_gradient
    def gradient(self, V):
        A = [
            g(g.qcd.gauge.project.traceless_anti_hermitian(u) / 1j)
            for u in g.qcd.gauge.transformed(self.U, V)
        ]
        dmuAmu = V.new()
        dmuAmu.otype = V.otype.cartesian()
        dmuAmu = g(0.0 * dmuAmu)
        for mu in range(3):  # Only spatial directions
            Amu = A[mu]
            dmuAmu += Amu - g.cshift(Amu, mu, -1)
        return dmuAmu