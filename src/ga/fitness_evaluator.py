# Directory: src/ga/fitness_evaluator.py

import numpy as np

class FitnessEvaluator:
    def __init__(self, solver, u_d, fields_ref=None, gamma=0):
        self.solver = solver
        self.u_d = u_d
        self.fields_ref = fields_ref or {}
        self.gamma = gamma

    def evaluate(self, fields_candidate):
        try:
            u = self.solver.solve(
                c=fields_candidate['c'],
                a=fields_candidate['a'],
                f=fields_candidate['f'],
                g=lambda x, y: 0.0
            )
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                return 1e6
            error = np.linalg.norm(u.flatten() - self.u_d.flatten()) / np.sqrt(u.size)
        except Exception:
            return 1e6

        reg = 0.0
        for field in ['c', 'a', 'f']:
            if field in self.fields_ref:
                reg += np.linalg.norm(fields_candidate[field] - self.fields_ref[field])
        
        return error + self.gamma * reg
