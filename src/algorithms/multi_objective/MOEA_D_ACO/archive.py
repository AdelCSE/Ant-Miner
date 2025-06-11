import numpy as np

def dominates(fitness1, fitness2):
    """Returns True if fitness1 dominates fitness2."""
    return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))

def update_EP(colony: dict, EP: list):
    """
    Updates the external Pareto archive (EP) with non-dominated solutions from the colony.

    Args:
        colony (dict): Should contain a list of ants under `colony["ant"]`, each with a `fitness` list.
        EP (list): Existing archive of non-dominated solutions (list of ant dicts).

    Returns:
        ant_best_rule (list[int]): Indices of ants that produced new non-dominated solutions.
        EP (list): Updated Pareto archive.
    """
    new_sol_EP = []
    ant_best_rule = []

    for i, ant in enumerate(colony["ant"]):
        dominated = False
        for ep_ant in EP:
            if dominates(ep_ant["fitness"], ant["fitness"]):
                dominated = True
                break

        if not dominated:
            # Remove all EP ants dominated by current ant
            EP = [ep_ant for ep_ant in EP if not dominates(ant["fitness"], ep_ant["fitness"])]

            # Add the new non-dominated solution
            EP.append(ant)
            new_sol_EP.append(ant)
            ant_best_rule.append(i)

    return ant_best_rule, EP