import numpy as np

class Individual:
    def __init__(self, shape, bounds, c0=None, a0=None, f0=None):
        # shape = (nx, ny) — size of each coefficient field
        self.low, self.high = bounds
        noise = 0.05  # 5% random offset

        # Init fields with optional references + noise
        self.c = (c0 if c0 is not None else np.ones(shape)) + np.random.uniform(-noise, noise, shape)
        self.a = (a0 if a0 is not None else np.zeros(shape)) + np.random.uniform(-noise, noise, shape)
        self.f = (f0 if f0 is not None else np.ones(shape)) + np.random.uniform(-noise, noise, shape)

        self._clip()

    def _clip(self):
        # Keep everything in allowed range
        self.c = np.clip(self.c, 0.5, 1.5)
        self.a = np.clip(self.a, -0.5, 0.5)
        self.f = np.clip(self.f, -10, 10)

    def crossover(self, other):
        # Average crossover (basic arithmetic)
        return Individual.from_fields(
            (self.c + other.c) * 0.5,
            (self.a + other.a) * 0.5,
            (self.f + other.f) * 0.5,
            self.low,
            self.high
        )

    def mutate(self, rate):
        # Slight tweaks to genes — f gets more aggressive mutation
        self.c += np.random.uniform(-rate, rate, self.c.shape)
        self.a += np.random.uniform(-rate, rate, self.a.shape)
        self.f += np.random.uniform(-10 * rate, 10 * rate, self.f.shape)
        self._clip()

    @staticmethod
    def from_fields(c, a, f, low, high):
        # Handy constructor from arrays
        ind = Individual(c.shape, (low, high))
        ind.c, ind.a, ind.f = c, a, f
        ind._clip()
        return ind

    def to_dict(self):
        # Dump fields for evaluator or solver
        return {"c": self.c, "a": self.a, "f": self.f}
