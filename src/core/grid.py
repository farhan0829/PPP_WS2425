import numpy as np

class Grid:
    def __init__(self, domain_size, grid_points):
        self.domain_size = domain_size
        self.grid_points = grid_points
        self.x, self.y, self.dx, self.dy = self._create_grid()

    def _create_grid(self):
        width, height = self.domain_size
        nx, ny = self.grid_points
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        dx = width / (nx - 1)
        dy = height / (ny - 1)
        return x, y, dx, dy

    def apply_boundary_conditions(self, u, value=0):
        u[0, :] = u[-1, :] = value
        u[:, 0] = u[:, -1] = value
        return u