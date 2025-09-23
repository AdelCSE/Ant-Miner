from .utils import get_gmax
from .decomposition import get_scalarized

def update_pheromone(colony, pheromones, best_ants, p, ant_groups, lambda_weights, eps, decomposition, reference, labels, task):

    if task == 'single':
        # Evaporation
        for group in pheromones:
            for term in pheromones[group]:
                pheromones[group][term] *= p

        # Deposit pheromone
        for ant_idx in best_ants:
            group = ant_groups[ant_idx]
            rule = colony['ants'][ant_idx]['rule']
            fitness = colony['ants'][ant_idx]["fitness"]
            weights = lambda_weights[ant_idx]

            scalarized = get_scalarized(fitness=fitness, reference=reference, weights=weights, approach=decomposition)

            for current_term in rule:
                if current_term[0] in pheromones[group]:
                    pheromones[group][current_term] += scalarized

        # Compute tau_max and tau_min using gmax
        B = len(best_ants)
        gmax = get_gmax(colony, lambda_weights, reference=reference, task=task, approach=decomposition)
        tau_max = ((B + 1) / ((1 - p))) * gmax if gmax != 0 else 1e6
        tau_min = eps * tau_max

        # Clamp pheromone values
        for group in pheromones:
            for term in pheromones[group]:
                pheromones[group][term] = max(tau_min, min(tau_max, pheromones[group][term]))

        return pheromones, tau_min, tau_max

    else:
        # Evaporation
        for group in pheromones:
            for label in pheromones[group]:
                for term in pheromones[group][label]:
                    pheromones[group][label][term] *= p

        # Deposit pheromone
        for ant_idx in best_ants:
            group = ant_groups[ant_idx]
            rules = colony['ants'][ant_idx]['ruleset']['rules']
            fitness = colony['ants'][ant_idx]['ruleset']["fitness"]
            weights = lambda_weights[ant_idx]

            scalarized = get_scalarized(fitness=fitness, reference=reference, weights=weights, approach=decomposition)

            for rule in rules:
                consequent = [term for term in rule if term[0] in labels]
                for label, _ in consequent:
                    for current_term in rule:
                        if current_term[0] not in labels and current_term[0] in pheromones[group].get(label, {}):
                            pheromones[group][label][current_term] += scalarized

        # Compute tau_max and tau_min using gmax
        B = len(best_ants)
        gmax = get_gmax(colony, lambda_weights, reference=reference, task=task, approach=decomposition)
        tau_max = ((B + 1) / ((1 - p))) * gmax if gmax != 0 else 1e6
        tau_min = eps * tau_max

        # Clamp pheromone values
        for group in pheromones:
            for label in pheromones[group]:
                for term in pheromones[group][label]:
                    pheromones[group][label][term] = max(tau_min, min(tau_max, pheromones[group][label][term]))

        return pheromones, tau_min, tau_max