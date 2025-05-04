# Directory: src/toolkit/fd_solver.py

import numpy as np

def assemble_fd(domain_dim, grid_shape, domain_size, c, a, f, bc, bc_type="dirichlet"):
    if domain_dim == 1:
        return _assemble_fd_1d(grid_shape[0], domain_size[0], c, a, f, bc, bc_type)
    elif domain_dim == 2:
        return _assemble_fd_2d(grid_shape, domain_size, c, a, f, bc, bc_type)
    else:
        raise NotImplementedError("Only 1D and 2D supported.")

def _assemble_fd_1d(nx, Lx, c, a, f, bc, bc_type):
    dx = Lx / (nx - 1)
    A = np.zeros((nx, nx))
    b = np.zeros(nx)

    for i in range(1, nx-1):
        A[i, i-1] = c[i] / dx**2
        A[i, i] = -2 * c[i] / dx**2 + a[i]
        A[i, i+1] = c[i] / dx**2
        b[i] = f[i]

    if bc_type == "dirichlet":
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0.0
        A[-1, :] = 0
        A[-1, -1] = 1
        b[-1] = 0.0
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return A, b

def _assemble_fd_2d(grid_shape, domain_size, c, a, f, bc, bc_type="dirichlet"):
    nx, ny = grid_shape
    Lx, Ly = domain_size
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    N = nx * ny
    A = np.zeros((N, N))
    b = np.zeros(N)

    def idx(i, j): return j * nx + i

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            k = idx(i, j)
            A[k, idx(i-1, j)] = c[i, j] / dx**2
            A[k, idx(i+1, j)] = c[i, j] / dx**2
            A[k, idx(i, j-1)] = c[i, j] / dy**2
            A[k, idx(i, j+1)] = c[i, j] / dy**2
            A[k, k] = -2 * (c[i, j] / dx**2 + c[i, j] / dy**2) + a[i, j]
            b[k] = f[i, j]

    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                A[k, :] = 0
                A[k, k] = 1
                b[k] = 0.0

    return A, b

def solve_fd(A, b):
    return np.linalg.solve(A, b)
