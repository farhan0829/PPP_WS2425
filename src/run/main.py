# Directory: src/run/main.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
from core.grid import Grid
from core.pde_solver import PDESolver
from problems.problem_setup import create_problem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fdm", "fem"], default="fdm", help="Select solving method")
    args = parser.parse_args()

    domain_size = (1.0, 1.0)
    grid_points = (50, 50)
    grid = Grid(domain_size, grid_points)
    u_d, a_d = create_problem(domain_size, grid_points)
    c = np.ones(grid_points)
    f = np.ones(grid_points)

    solver = PDESolver(grid)
    u = solver.solve(c, a_d, f, method=args.method)

    x = np.linspace(0, domain_size[0], grid_points[0])
    y = np.linspace(0, domain_size[1], grid_points[1])
    X, Y = np.meshgrid(x, y)

    plt.contourf(X, Y, u)
    plt.title(f"Solution using {args.method.upper()}")
    plt.colorbar()
    plt.show()
