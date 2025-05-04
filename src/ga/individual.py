# Directory: src/ga/individual.py
# -----------------------------------------------------------
#  Individual = GA chromosome holding three coefficient fields
#  ----------------------------------------------------------
#  * c : diffusion coefficient   (clipped to 0.5 … 1.5)
#  * a : reaction coefficient    (clipped to –0.5 … 0.5)
#  * f : source‑term             (clipped to –10 … 10)
#
#  The class supports:
#  • random initialisation around optional reference fields
#  • uniform arithmetic crossover
#  • bounded mutation with tunable rate
#  • conversion <‑‑> dict for the FitnessEvaluator / solver
# -----------------------------------------------------------

import numpy as np

class Individual:
    def __init__(
        self,
        grid_shape,
        bounds,
        c_ref=None,
        a_ref=None,
        f_ref=None
    ):
        """
        Parameters
        ----------
        grid_shape : tuple
            Spatial grid shape, e.g. (nx,) or (nx, ny).
        bounds : (float, float)
            Global (lower, upper) bound used only for initial random noise.
        c_ref, a_ref, f_ref : ndarray or None
            Optional reference fields.  If None, defaults are
            c=1, a=0, f=1 with added ±5 % noise.
        """
        self.lower, self.upper = bounds
        noise = 0.05  # ±5 % random perturbation

        # initialise fields around references (or defaults)
        self.c = (c_ref if c_ref is not None else np.ones(grid_shape)) \
                 + np.random.uniform(-noise,  noise,  grid_shape)
        self.a = (a_ref if a_ref is not None else np.zeros(grid_shape)) \
                 + np.random.uniform(-noise,  noise,  grid_shape)
        self.f = (f_ref if f_ref is not None else np.ones(grid_shape)) \
                 + np.random.uniform(-noise,  noise,  grid_shape)

        self.clip_fields()

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def clip_fields(self):
        """Project all gene values back to their admissible ranges."""
        self.c = np.clip(self.c,  0.5,   1.5)
        self.a = np.clip(self.a, -0.5,   0.5)
        self.f = np.clip(self.f, -10.0, 10.0)

    # --------------------------------------------------------
    # genetic operators
    # --------------------------------------------------------
    def crossover(self, other):
        """
        Uniform arithmetic crossover (50 / 50 average).

        Returns
        -------
        Individual
            Child with fields = (parent1 + parent2) / 2
        """
        return Individual.from_fields(
            (self.c + other.c) / 2.0,
            (self.a + other.a) / 2.0,
            (self.f + other.f) / 2.0,
            self.lower,
            self.upper
        )

    def mutate(self, mutation_rate):
        """
        Add uniform noise and re‑clip.

        Parameters
        ----------
        mutation_rate : float
            Magnitude of mutation; applied as ±rate for c and a,
            and ±10·rate for f (larger dynamic range).
        """
        self.c += np.random.uniform(-mutation_rate,        mutation_rate,        self.c.shape)
        self.a += np.random.uniform(-mutation_rate,        mutation_rate,        self.a.shape)
        self.f += np.random.uniform(-mutation_rate * 10.0, mutation_rate * 10.0, self.f.shape)
        self.clip_fields()

    # --------------------------------------------------------
    # factory utility
    # --------------------------------------------------------
    @staticmethod
    def from_fields(c, a, f, lower, upper):
        """
        Construct an Individual directly from explicit field arrays.
        """
        obj = Individual(c.shape, (lower, upper))
        obj.c, obj.a, obj.f = c, a, f
        obj.clip_fields()
        return obj

    # --------------------------------------------------------
    # convenience
    # --------------------------------------------------------
    def to_dict(self):
        """Return fields as a dict → convenient for solver / evaluator."""
        return {"c": self.c, "a": self.a, "f": self.f}
