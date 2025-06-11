import numpy as np

def update_pheromone(pheromones, colony, best_ants_indices, p, ant_groups, lambda_weights, eps):
    nb_ants = len(colony["ant"])

    # Evaporation
    for group in pheromones:
        for term in pheromones[group]:
            pheromones[group][term] *= p

    # Deposit pheromone
    for ant_idx in best_ants_indices:
        group = ant_groups[ant_idx]
        rule = colony["ant"][ant_idx]["rule"]
        fitness = colony["ant"][ant_idx]["fitness"]
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

def find_gmax(colony, lambda_weights, start, end):
    gmax = float("-inf")
    for i in range(start - 1, end):
        weights = lambda_weights[i]
        fitness = colony["ant"][i]["fitness"]
        scalarized = sum(w * f for w, f in zip(weights, fitness))
        if scalarized > gmax:
            gmax = scalarized
    return gmax
