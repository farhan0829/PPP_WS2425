# Directory: src/testcases/test_fd_solver_with_ga.py
import time
import numpy as np
import matplotlib.pyplot as plt
from toolkit.fd_solver import assemble_fd, solve_fd
from problems.elliptic_solver import FEMEllipticSolver
from optimizer.ga_optimizer import GAOptimizer

def solve_fem(c, a, f):
    nx = c.shape[0] - 1 if c.ndim == 1 else c.shape[0] - 1
    ny = 0 if c.ndim == 1 else c.shape[1] - 1
    dx = 1.0 / (nx if nx > 0 else 1)
    dy = 1.0 / (ny if ny > 0 else 1)
    solver = FEMEllipticSolver(nx, ny, dx, dy)
    return solver.solve(c, a, f, g=lambda x, y: 0.0, interpolate=False)

def test_1d_dirichlet_with_ga():
    Lx = 1.0
    grid_sizes = [40, 80, 100]
    pop_size = 30
    mutation_rate = 0.005
    generations = 300

    print("--- 1D Dirichlet Convergence with GA ---")
    for nx in grid_sizes:
        x = np.linspace(0, Lx, 2 * nx + 1)
        u_exact = x * (1 - x)

        f = np.full((1, 1), 2.0)
        c = np.ones((1, 1))
        a = np.zeros((1, 1))

        solver_fn = lambda **fields: solve_fem(fields.get('c', c), fields.get('a', a), fields.get('f', f))
        u_fem_before = solver_fn()
        error_before = np.linalg.norm(u_fem_before.flatten() - u_exact.flatten(), ord=2) / np.sqrt(2 * nx + 1)

        ga = GAOptimizer(pop_size=pop_size,
                         mutation_rate=mutation_rate,
                         generations=generations,
                         grid_shape=(1, 1),
                         solver_fn=solver_fn,
                         u_exact=u_exact,
                         optimize_fields=("f",))
        best_fields = ga.optimize()

        u_fem_after = solve_fem(c, a, best_fields['f'].reshape(1, 1))
        error_after = np.linalg.norm(u_fem_after.flatten() - u_exact.flatten(), ord=2) / np.sqrt(2 * nx + 1)

        print(f"[nx={nx}] FEM L2 Error Before GA: {error_before:.2e}, After GA: {error_after:.2e}")

def test_2d_dirichlet_with_ga():
    Lx, Ly = 1.0, 1.0
    nx, ny = 50, 50 
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_exact = (X * (1 - X)) * (Y * (1 - Y))
    f = np.full_like(X, 4.0)
    c = np.ones_like(f)
    a = np.zeros_like(f)

    solver_fn = lambda **fields: solve_fem(fields.get('c', c.T), fields.get('a', a.T), fields.get('f', f.T)).T
    u_fem_before = solver_fn()
    error_before = np.max(np.abs(u_fem_before - u_exact))

    ga = GAOptimizer(pop_size=30,
                     mutation_rate=0.005,
                     generations=500,
                     grid_shape=(nx, ny),
                     solver_fn=solver_fn,
                     u_exact=u_exact,
                     optimize_fields=("f",))

    best_fields = ga.optimize()

    u_fem_after = solve_fem(c.T, a.T, best_fields['f'].T).T
    error_after = np.max(np.abs(u_fem_after - u_exact))

    print("--- 2D Dirichlet with GA ---")
    print(f"[2D FEM] Max Error Before GA: {error_before:.2e}")
    print(f"[2D FEM] Max Error After  GA: {error_after:.2e}")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u_fem_before, levels=50)
    plt.title("2D FEM Before GA")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u_fem_after, levels=50)
    plt.title("2D FEM After GA")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, np.abs(u_fem_after - u_exact), levels=50)
    plt.title("Absolute Error After GA")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()

    print("Starting full solver test suite...\n")
    t1 = time.time()
    test_1d_dirichlet_with_ga()
    t2 = time.time()
    print(f"\n[Time] 1D Dirichlet Tests Completed using GA in {t2 - t1:.2f} seconds.")

    t3 = time.time()
    test_2d_dirichlet_with_ga()
    t4 = time.time()
    print(f"\n[Time] 2D Dirichlet Tests Completed using GA in {t4 - t3:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"\n✅ Total execution time: {total_time:.2f} seconds.")
