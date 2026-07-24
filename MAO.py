import numpy as np
import time


def evaluate_fitness(population, fitness_function):
    return np.array([fitness_function(ind) for ind in population])

def mexican_hat_wave(x, A=1, B=1, C=0, D=0):
    return A * (1 - x**2) * np.exp(-B * x**2) + C * (1 - D * x**2) * np.exp(-D * x**2)

def MAO(population, fitness_function, lb, ub, max_iterations):
    population_size, dimension = population.shape
    best_fitness = np.inf
    best_solution = None

    Convergence_curve = np.zeros((1, max_iterations))
    ct = time.time()

    for i in range(max_iterations):
        fitness = evaluate_fitness(population, fitness_function)
        min_fitness_idx = np.argmin(fitness)

        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = population[min_fitness_idx]

        hat_wave = mexican_hat_wave(np.random.uniform(-1, 1, size=(population_size, dimension)))
        new_solutions = population + hat_wave
        new_solutions = np.clip(new_solutions, lb, ub)

        population = new_solutions
        Convergence_curve[0, i] = min(best_solution)
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct

