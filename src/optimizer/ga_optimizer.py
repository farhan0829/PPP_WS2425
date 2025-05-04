# Directory: src/optimizer/ga_optimizer.py

import numpy as np
import matplotlib.pyplot as plt

class GAOptimizer:
    def __init__(self, pop_size, mutation_rate, generations, grid_shape, solver_fn, u_exact, optimize_fields=("f",)):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.grid_shape = grid_shape  # Can be (nx,) or (nx, ny)
        self.solver_fn = solver_fn
        self.u_exact = u_exact
        self.optimize_fields = optimize_fields

    def initialize_population(self):
        num_params = len(self.optimize_fields)
        return np.random.randn(self.pop_size, *self.grid_shape, num_params)

    def evaluate_fitness(self, population):
        fitness = np.zeros(self.pop_size)
        for i, candidate in enumerate(population):
            kwargs = {}
            for idx, field in enumerate(self.optimize_fields):
                field_data = candidate[..., idx]
                if field == "c":
                    field_data = np.clip(field_data, 0.5, 2.0)  # ensure c > 0
                if field == "a":
                    field_data = np.clip(field_data, -1.0, 1.0)
                if field == "f":
                    field_data = np.clip(field_data, -10.0, 10.0)
                kwargs[field] = field_data

            try:
                u_numeric = self.solver_fn(**kwargs)
                if np.any(np.isnan(u_numeric)) or np.any(np.isinf(u_numeric)):
                    fitness[i] = 1e6
                else:
                    fitness[i] = np.linalg.norm(u_numeric.flatten() - self.u_exact.flatten(), ord=2) / np.sqrt(u_numeric.size)
            except Exception:
                fitness[i] = 1e6
        return fitness

    def select_parents(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx[:2]]

    def crossover(self, parents):
        child = np.copy(parents[0])
        mask = np.random.rand(*child.shape) > 0.5
        child[mask] = parents[1][mask]
        return child

    def mutate(self, individual):
        mutation = self.mutation_rate * np.random.randn(*individual.shape)
        return individual + mutation

    def optimize(self):
        population = self.initialize_population()
        history = []

        for gen in range(self.generations):
            fitness = self.evaluate_fitness(population)
            best_idx = np.argmin(fitness)
            best_fit = fitness[best_idx]
            history.append(best_fit)

            print(f"[Gen {gen+1}] Best L2 Error: {best_fit:.2e}")

            new_population = []
            for _ in range(self.pop_size):
                parents = self.select_parents(population, fitness)
                child = self.crossover(parents)
                child = self.mutate(child)
                new_population.append(child)
            population = np.array(new_population)

        final_fitness = self.evaluate_fitness(population)
        best_idx = np.argmin(final_fitness)
        best_solution = population[best_idx]

        plt.figure()
        plt.plot(history, 'o-')
        plt.yscale('log')
        plt.xlabel("Generation")
        plt.ylabel("L2 Error")
        plt.title("GA Convergence Plot")
        plt.grid()
        plt.show()

        # Split the best solution fields
        best_fields = {}
        for idx, field in enumerate(self.optimize_fields):
            best_fields[field] = best_solution[..., idx]
        return best_fields
