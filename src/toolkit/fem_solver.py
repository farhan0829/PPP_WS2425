# Directory: src/toolkit/fem_solver.py

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def assemble_fem(domain_dim, grid_shape, domain_size, c, a, f, bc_dirichlet):
    if domain_dim == 1:
        return _assemble_fem_1d(grid_shape[0], domain_size[0], c, a, f, bc_dirichlet)
    elif domain_dim == 2:
        return _assemble_fem_2d(grid_shape, domain_size, c, a, f, bc_dirichlet)
    else:
        raise NotImplementedError("Only 1D and 2D supported.")

def _assemble_fem_1d(nx, Lx, c, a, f, bc_dirichlet):
    dx = Lx / (nx - 1)
    A = lil_matrix((nx, nx))
    b = np.zeros(nx)

    for i in range(nx - 1):
        nodes = [i, i + 1]
        c_local = (c[i] + c[i+1]) / 2
        a_local = (a[i] + a[i+1]) / 2
        f_local = (f[i] + f[i+1]) / 2

        Ke = (c_local / dx) * np.array([[1, -1], [-1, 1]])
        Me = (a_local * dx / 6) * np.array([[2, 1], [1, 2]])
        fe = (f_local * dx / 2) * np.array([1, 1])

        A[nodes[0], nodes[0]] += Ke[0, 0] + Me[0, 0]
        A[nodes[0], nodes[1]] += Ke[0, 1] + Me[0, 1]
        A[nodes[1], nodes[0]] += Ke[1, 0] + Me[1, 0]
        A[nodes[1], nodes[1]] += Ke[1, 1] + Me[1, 1]

        b[nodes[0]] += fe[0]
        b[nodes[1]] += fe[1]

    # Apply Dirichlet BCs
    A[0, :] = 0
    A[0, 0] = 1
    b 

    A[-1, :] = 0
    A[-1, -1] = 1
    b[-1] = bc_dirichlet['right'](Lx)

    return A.tocsr(), b

def _assemble_fem_2d(grid_shape, domain_size, c, a, f, bc_dirichlet):
    nx, ny = grid_shape
    Lx, Ly = domain_size
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    num_nodes = nx * ny
    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    def idx(i, j):
        return j * nx + i

    # Gauss points and weights
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    w = [1, 1]

    for j in range(ny - 1):
        for i in range(nx - 1):
            nodes = [
                idx(i, j),
                idx(i+1, j),
                idx(i+1, j+1),
                idx(i, j+1)
            ]

            x0, y0 = i * dx, j * dy

            Ke = np.zeros((4, 4))
            fe = np.zeros(4)

            for xi, wx in zip(gp, w):
                for eta, wy in zip(gp, w):
                    N = 0.25 * np.array([
                        (1 - xi) * (1 - eta),
                        (1 + xi) * (1 - eta),
                        (1 + xi) * (1 + eta),
                        (1 - xi) * (1 + eta)
                    ])

                    dN_dxi = np.array([
                        [-(1 - eta), -(1 - xi)],
                        [ (1 - eta), -(1 + xi)],
                        [ (1 + eta),  (1 + xi)],
                        [-(1 + eta),  (1 - xi)]
                    ]) * 0.25

                    J = np.array([[dx/2, 0], [0, dy/2]])
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)
                    gradN = dN_dxi @ invJ

                    x_phys = x0 + dx * (xi + 1) / 2
                    y_phys = y0 + dy * (eta + 1) / 2

                    c_val = (c[i, j] + c[i+1, j] + c[i+1, j+1] + c[i, j+1]) / 4
                    a_val = (a[i, j] + a[i+1, j] + a[i+1, j+1] + a[i, j+1]) / 4
                    f_val = (f[i, j] + f[i+1, j] + f[i+1, j+1] + f[i, j+1]) / 4

                    Ke += (c_val * (gradN @ gradN.T) + a_val * np.outer(N, N)) * detJ * wx * wy
                    fe += N * f_val * detJ * wx * wy

            for m in range(4):
                for n in range(4):
                    A[nodes[m], nodes[n]] += Ke[m, n]
                b[nodes[m]] += fe[m]

    # Apply Dirichlet BCs
    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)
            if i == 0:
                A[k, :] = 0; A[k, k] = 1; b[k] = bc_dirichlet['left'](j*dy)
            if i == nx - 1:
                A[k, :] = 0; A[k, k] = 1; b[k] = bc_dirichlet['right'](j*dy)
            if j == 0:
                A[k, :] = 0; A[k, k] = 1; b[k] = bc_dirichlet['bottom'](i*dx)
            if j == ny - 1:
                A[k, :] = 0; A[k, k] = 1; b[k] = bc_dirichlet['top'](i*dx)

    return A.tocsr(), b

def solve_fem(A, b):
    return spsolve(A, b)
 