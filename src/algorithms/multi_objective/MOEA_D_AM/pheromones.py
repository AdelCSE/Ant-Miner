import numpy as np
from .utils import find_gmax

def update_pheromone(pheromones, colony, best_ants_indices, p, ant_groups, lambda_weights, eps):
    nb_ants = len(colony['ants'])

    # Evaporation
    for group in pheromones:
        for term in pheromones[group]:
            pheromones[group][term] *= p


    # Deposit pheromone
    for ant_idx in best_ants_indices:
        group = ant_groups[ant_idx]
        rule = colony['ants'][ant_idx]["rule"]
        fitness = colony['ants'][ant_idx]["fitness"]
        weights = lambda_weights[ant_idx]

        scalarized = sum(weights[l] * fitness[l] for l in range(len(fitness)))
        for current_term in rule:
            if current_term[0] != "class":
                pheromones[group][current_term] *= scalarized

    # Compute tau_max and tau_min using gmax
    B = len(best_ants_indices)
    gmax = find_gmax(colony, lambda_weights, 1, nb_ants)
    tau_max = ((B + 1) / ((1 - p))) * gmax if gmax != 0 else 1e6
    tau_min = eps * tau_max

    # Clamp pheromone values
    for group in pheromones:
        for term in pheromones[group]:
            pheromones[group][term] = max(tau_min, min(tau_max, pheromones[group][term]))

    return pheromones, tau_min, tau_max