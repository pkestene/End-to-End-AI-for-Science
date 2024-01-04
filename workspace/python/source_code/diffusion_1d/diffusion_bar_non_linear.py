# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import numpy as np
from sympy import Symbol, Eq, Function, Number

import modulus
from modulus.sym.hydra import instantiate_arch , ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.eq.pde import PDE

import sympy as sympy

# params for domain
x0 = 0
x1 = 1
L1 = Line1D(x0, x1)

D0 = 1e1
D1 = 1e-1

T0 = 100
T1 = 10

print(T0)
print(T1)

# constant used in the analytical solution
A = (D1-D0) * (T0-T1) / np.log(D0/D1)
B = T0 - (T0-T1) / np.log(D0/D1) * np.log(D0)

class Diffusion(PDE):
    name = "Diffusion"

    def __init__(self, T="T", D0=1, D1=2, Q=0, dim=3, time=True):
        # set params
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)

        # Diffusivity
        D0 = Number(D0)
        D1 = Number(D1)
        D = D0 + x*(D1-D0)

        # Source
        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # set equations
        self.equations = {}
        self.equations["diffusion_" + self.T] = (
            T.diff(t)
            - ( (D0+x*(D1-D0)) * T.diff(x)).diff(x)
            - Q
        )


@modulus.sym.main(config_path="conf", config_name="config_non_linear")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    diff_u = Diffusion(T="u", D0=D0, D1=D1, dim=1, time=False)

    diff_net_u = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        diff_u.make_nodes()
        + [diff_net_u.make_node(name="u_network", jit=cfg.jit)]
    )

    # make domain add constraints to the solver
    domain = Domain()

    # sympy variables
    x = Symbol("x")

    # left hand side (x = x0)
    lhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"u": T0},
        batch_size=cfg.batch_size.lhs,
        criteria=Eq(x, x0),
    )
    domain.add_constraint(lhs, "left_hand_side")

    # right hand side (x = x1)
    rhs = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"u": T1},
        batch_size=cfg.batch_size.rhs,
        criteria=Eq(x, x1),
    )
    domain.add_constraint(rhs, "right_hand_side")

    # interior
    interior_u = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=L1,
        outvar={"diffusion_u": 0},
        bounds={x: (x0, x1)},
        batch_size=cfg.batch_size.interior_u,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "diffusion_u": Symbol("sdf")
        }
    )
    domain.add_constraint(interior_u, "interior_u")

    # validation data with analytical solution
    x = np.expand_dims(np.linspace(x0, x1, 100), axis=-1)
    u = A / (D1-D0) * np.log(D0+x*(D1-D0)) + B
    invar_numpy = {"x": x}
    outvar_numpy = {"u": u}
    val = PointwiseValidator(nodes=nodes,invar=invar_numpy, true_outvar=outvar_numpy)
    domain.add_validator(val, name="Values")

    # make monitors
    invar_numpy = {"x": [[0.5]]}
    monitor = PointwiseMonitor(
        invar_numpy,
        output_names=["u__x"],
        metrics={"flux_u": lambda var: torch.mean(var["u__x"])},
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
