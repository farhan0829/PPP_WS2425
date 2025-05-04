# Directory: src/ga/genetic_algorithm.py

from .population import Population

class GeneticAlgorithm:
    def __init__(self, evaluator, grid_shape, c_ref=None, a_ref=None, f_ref=None,
                 bounds=(0.5, 1.5), population_size=30, generations=50, top_k=5, mutation_rate=0.05):
        self.evaluator = evaluator
        self.grid_shape = grid_shape
        self.bounds = bounds
        self.population = Population(population_size, grid_shape, bounds, c_ref, a_ref, f_ref)
        self.generations = generations
        self.top_k = top_k
        self.mutation_rate = mutation_rate

    def run(self):
        best_individual = None
        for gen in range(self.generations):
            best_individual = self.population.evolve(self.evaluator, self.top_k, self.mutation_rate)
            best_score = self.evaluator.evaluate(best_individual.to_dict())
            print(f"Generation {gen+1}: Best Score = {best_score:.4e}")
        return best_individual.to_dict()
