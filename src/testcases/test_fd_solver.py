# Directory: src/testcases/test_fd_solver.py
import time
import numpy as np
import matplotlib.pyplot as plt
from toolkit.fd_solver import assemble_fd, solve_fd
from problems.elliptic_solver import FEMEllipticSolver

def test_1d_dirichlet():
    Lx = 1.0
    grid_sizes = [20, 40, 80, 160]
    h_vals, fdm_errors, fem_errors = [], [], []

    print("--- 1D Dirichlet Convergence ---")
    for nx in grid_sizes:
        x = np.linspace(0, Lx, 2*nx+1)
        dx = Lx / nx
        h_vals.append(dx)

        u_exact = x * (1 - x)
        f = np.full((1,1), 2.0)
        c = np.ones((1,1))
        a = np.zeros((1,1))

        bc = {'left': lambda x: 0.0, 'right': lambda x: 0.0}

        A_fd, b_fd = assemble_fd(1, (2*nx+1,), (Lx,), np.ones_like(x), np.zeros_like(x), np.full_like(x,2.0), bc, bc_type='dirichlet')
        u_fd = solve_fd(A_fd, b_fd)
        error_fd = np.linalg.norm(u_fd - u_exact, ord=2) / np.sqrt(2*nx+1)

        fem_solver = FEMEllipticSolver(nx, 0, dx/2, 1.0)
        u_fem = fem_solver.solve(c, a, f, g=lambda x,y: 0.0, interpolate=False).flatten()
        error_fem = np.linalg.norm(u_fem - u_exact, ord=2) / np.sqrt(2*nx+1)

        print(f"[nx={nx}] FDM L2 Error: {error_fd:.2e}, FEM L2 Error: {error_fem:.2e}")
        fdm_errors.append(error_fd)
        fem_errors.append(error_fem)

    plt.figure()
    plt.loglog(h_vals, fdm_errors, 'o-', label='FDM')
    plt.loglog(h_vals, fem_errors, 's--', label='FEM')
    plt.gca().invert_xaxis()
    plt.grid(True, which='both')
    plt.xlabel('Grid spacing h')
    plt.ylabel('L2 Error')
    plt.title('1D Dirichlet Convergence (FDM vs FEM)')
    plt.legend()
    plt.show()

def test_2d_dirichlet():
    Lx, Ly = 1.0, 1.0
    nx, ny = 100, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_exact = (X*(1-X)) * (Y*(1-Y))
    f = np.full_like(X, 4.0)
    c = np.ones_like(f)
    a = np.zeros_like(f)

    bc = {'left': lambda y: 0.0, 'right': lambda y: 0.0, 'bottom': lambda x: 0.0, 'top': lambda x: 0.0}

    A_fd, b_fd = assemble_fd(2, (nx, ny), (Lx, Ly), c, a, f, bc)
    u_fd = solve_fd(A_fd, b_fd).reshape((nx, ny))

    fem_solver = FEMEllipticSolver(nx-1, ny-1, Lx/(nx-1), Ly/(ny-1))
    u_fem = fem_solver.solve(c.T.copy(), a.T.copy(), f.T.copy(), g=lambda x,y: 0.0, interpolate=False).T

    error_fd = np.max(np.abs(u_fd - u_exact))
    error_fem = np.max(np.abs(u_fem - u_exact))

    print("--- 2D Dirichlet ---")
    print(f"[2D FDM] Max Error: {error_fd:.2e}")
    print(f"[2D FEM] Max Error: {error_fem:.2e}")

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.contourf(X, Y, u_fd, levels=50)
    plt.title("2D FDM Solution")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.contourf(X, Y, u_fem, levels=50)
    plt.title("2D FEM Solution")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.contourf(X, Y, np.abs(u_fem - u_exact), levels=50)
    plt.title("2D FEM Absolute Error")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()

    print("Starting full solver test suite...\n")
    t1 = time.time()
    test_1d_dirichlet()
    t2 = time.time()
    print(f"\n[Time] 1D Dirichlet Tests Completed in {t2 - t1:.2f} seconds.")

    t3 = time.time()
    test_2d_dirichlet()
    t4 = time.time()
    print(f"\n[Time] 2D Dirichlet Tests Completed in {t4 - t3:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"\n✅ Total execution time: {total_time:.2f} seconds.")
