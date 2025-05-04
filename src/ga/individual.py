# Directory: src/ga/individual.py

import numpy as np

class Individual:
    def __init__(self, grid_shape, bounds, c_ref=None, a_ref=None, f_ref=None):
        self.lower, self.upper = bounds
        noise_level = 0.05  # 5% noise
        
        self.c = (c_ref if c_ref is not None else np.ones(grid_shape)) + np.random.uniform(-noise_level, noise_level, grid_shape)
        self.a = (a_ref if a_ref is not None else np.zeros(grid_shape)) + np.random.uniform(-noise_level, noise_level, grid_shape)
        self.f = (f_ref if f_ref is not None else np.ones(grid_shape)) + np.random.uniform(-noise_level, noise_level, grid_shape)

        self.clip_fields()

    def clip_fields(self):
        self.c = np.clip(self.c, 0.5, 1.5)
        self.a = np.clip(self.a, -0.5, 0.5)
        self.f = np.clip(self.f, -10.0, 10.0)

    def crossover(self, other):
        child = Individual.from_fields(
            (self.c + other.c) / 2,
            (self.a + other.a) / 2,
            (self.f + other.f) / 2,
            self.lower, self.upper
        )
        return child

    def mutate(self, mutation_rate):
        self.c += np.random.uniform(-mutation_rate, mutation_rate, self.c.shape)
        self.a += np.random.uniform(-mutation_rate, mutation_rate, self.a.shape)
        self.f += np.random.uniform(-mutation_rate*10, mutation_rate*10, self.f.shape)
        self.clip_fields()

    @staticmethod
    def from_fields(c, a, f, lower, upper):
        obj = Individual(c.shape, (lower, upper))
        obj.c, obj.a, obj.f = c, a, f
        obj.clip_fields()
        return obj

    def to_dict(self):
        return {"c": self.c, "a": self.a, "f": self.f}
