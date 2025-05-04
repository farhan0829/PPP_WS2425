# Directory: src/problems/elliptic_solver.py

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class FEMEllipticSolver:
    def __init__(self, Nx, Ny, dx, dy):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        if Ny == 0:
            self.num_nodes = 2 * Nx + 1  # Quadratic elements (P2)
        else:
            self.num_nodes = (Nx + 1) * (Ny + 1)  # 2D linear elements (Q4)

    def node_index(self, i, j):
        return j * (self.Nx + 1) + i

    def assemble(self, c, a, f, interpolate=False, neumann_bc=None):
        A = lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)

        if self.Ny == 0:
            # 1D FEM assembly
            f_value = f.flatten()[0]
            c_value = c.flatten()[0]
            a_value = a.flatten()[0]

            for i in range(self.Nx):
                nodes = [i, i+1]
                h = self.dx

                k_local = (c_value / h) * np.array([[1, -1], [-1, 1]]) + (a_value * h / 6) * np.array([[2, 1], [1, 2]])
                b_local = (f_value * h / 2) * np.array([1, 1])

                for m in range(2):
                    for n in range(2):
                        A[nodes[m], nodes[n]] += k_local[m, n]
                    b[nodes[m]] += b_local[m]

            if neumann_bc:
                if 'left' in neumann_bc:
                    b[0] += neumann_bc['left'](0.0)
                if 'right' in neumann_bc:
                    b[-1] += neumann_bc['right'](self.dx * self.Nx)

        else:
            # 2D FEM assembly (Q4 elements)
            gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
            weights = [1, 1]

            for j in range(self.Ny):
                for i in range(self.Nx):
                    nodes = [
                        self.node_index(i, j),
                        self.node_index(i+1, j),
                        self.node_index(i+1, j+1),
                        self.node_index(i, j+1)
                    ]
                    Ke = np.zeros((4, 4))
                    be = np.zeros(4)

                    for xi, wx in zip(gp, weights):
                        for eta, wy in zip(gp, weights):
                            N = 0.25 * np.array([
                                (1 - xi) * (1 - eta),
                                (1 + xi) * (1 - eta),
                                (1 + xi) * (1 + eta),
                                (1 - xi) * (1 + eta)
                            ])
                            dN_dxi = 0.25 * np.array([
                                [-(1 - eta), -(1 - xi)],
                                [ (1 - eta), -(1 + xi)],
                                [ (1 + eta),  (1 + xi)],
                                [-(1 + eta),  (1 - xi)]
                            ])

                            J = np.array([[self.dx/2, 0], [0, self.dy/2]])
                            detJ = np.linalg.det(J)
                            invJ = np.linalg.inv(J)
                            gradN = dN_dxi @ invJ.T

                            ci = c[j, i] if c.ndim == 2 else c.flatten()[0]
                            ai = a[j, i] if a.ndim == 2 else a.flatten()[0]
                            fi = f[j, i] if f.ndim == 2 else f.flatten()[0]

                            Ke += (gradN @ gradN.T) * ci * detJ * wx * wy + ai * np.outer(N, N) * detJ * wx * wy
                            be += N * fi * detJ * wx * wy

                    for m in range(4):
                        for n in range(4):
                            A[nodes[m], nodes[n]] += Ke[m, n]
                        b[nodes[m]] += be[m]

        return A, b

    def apply_boundary_conditions(self, A, b, g=None):
        if self.Ny == 0:
            for i in [0, -1]:
                A[i, :] = 0
                A[i, i] = 1
                b[i] = 0.0
        else:
            for j in range(self.Ny + 1):
                for i in range(self.Nx + 1):
                    if i == 0 or i == self.Nx or j == 0 or j == self.Ny:
                        k = self.node_index(i, j)
                        A[k, :] = 0
                        A[k, k] = 1
                        b[k] = 0.0
        return A, b

    def solve(self, c, a, f, g=None, interpolate=False, neumann_bc=None):
        A, b = self.assemble(c, a, f, interpolate=interpolate, neumann_bc=neumann_bc)
        A, b = self.apply_boundary_conditions(A, b, g=g)
        u = spsolve(A.tocsr(), b)
        return u.reshape((self.Ny+1, self.Nx+1)) if self.Ny > 0 else u
