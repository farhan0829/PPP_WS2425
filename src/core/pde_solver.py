# Directory: src/core/pde_solver.py

import numpy as np
from toolkit.fd_solver import assemble_fd, solve_fd
from toolkit.fem_solver import assemble_fem, solve_fem

class PDESolver:
    def __init__(self, grid):
        self.grid = grid

    def solve(self, c, a, f, method="fdm", **kwargs):
        nx, ny = self.grid.grid_points
        Lx, Ly = self.grid.domain_size
        bc = {
            'left': lambda x: 0,
            'right': lambda x: 0,
            'bottom': lambda x: 0,
            'top': lambda x: 0,
        }

        if method == "fdm":
            A, b = assemble_fd(
                domain_dim=2, grid_shape=(nx, ny), domain_size=(Lx, Ly),
                c=c, a=a, f=f, bc=bc
            )
            u = solve_fd(A, b).reshape((nx, ny))
        elif method == "fem":
            A, b = assemble_fem(
                domain_dim=2, grid_shape=(nx, ny), domain_size=(Lx, Ly),
                c=c, a=a, f=f, bc_dirichlet=bc
            )
            u = solve_fem(A, b).reshape((nx, ny))
        else:
            raise ValueError(f"Unknown method {method}")
        return u
