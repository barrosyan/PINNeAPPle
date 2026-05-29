# SPDX-License-Identifier: Apache-2.0
"""
Wave equation 1D
u_tt - (c^2 u_x)_x = 0
"""
from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE


class WaveEquation1D(PDE):
    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        x = Symbol("x")
        t = Symbol("t")
        input_variables = {"x": x, "t": t}
        u = Function("u")(*input_variables)

        if isinstance(c, str):
            c = Function(c)(*input_variables)
        elif isinstance(c, (float, int)):
            c = Number(c)

        self.equations = {}
        self.equations["wave_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)