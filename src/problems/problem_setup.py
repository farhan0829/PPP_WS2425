import numpy as np

def create_problem(domain_size=(1.0, 1.0), grid_points=(10, 10)):
    nx, ny = grid_points
    x = np.linspace(0, domain_size[0], nx)
    y = np.linspace(0, domain_size[1], ny)
    X, Y = np.meshgrid(x, y)
    u_d = np.sin(np.pi * X) * np.sin(np.pi * Y)
    a_d = np.ones_like(u_d)
    return u_d, a_d