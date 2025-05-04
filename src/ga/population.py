# Directory: src/ga/population.py

from .individual import Individual
import random

class Population:
    def __init__(self, size, grid_shape, bounds, c_ref=None, a_ref=None, f_ref=None):
        self.individuals = [Individual(grid_shape, bounds, c_ref, a_ref, f_ref) for _ in range(size)]

    def evolve(self, evaluator, top_k, mutation_rate):
        scores = [(ind, evaluator.evaluate(ind.to_dict())) for ind in self.individuals]
        scores.sort(key=lambda x: x[1])
        top_individuals = [ind for ind, _ in scores[:top_k]]
        new_population = []

        while len(new_population) < len(self.individuals):
            p1, p2 = random.sample(top_individuals, 2)
            child = p1.crossover(p2)
            child.mutate(mutation_rate)
            new_population.append(child)

        self.individuals = new_population
        return scores[0][0]  # Best individual
