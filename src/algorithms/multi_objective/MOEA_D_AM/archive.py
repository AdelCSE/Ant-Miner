def dominates(fitness1: list, fitness2: list) -> bool:
    """
    Returns True if fitness1 dominates fitness2.
    """
    return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))


def update_archive(colony: dict, archive: list, positive_class: str, rulesets: str) -> tuple:
    """
    Updates the external Pareto archive with non-dominated solutions from the colony.

    Args:
        colony (dict): Colony containing ants with their rules and fitness.
        archive (list): Existing archive of non-dominated solutions.

    Returns:
        best_ants_indices (list[int]): Indices of the best ants that contributed to the archive.
        archive (list): List of non-dominated solutions after the update.
    """

    ruleset_size = 2
    best_ants_indices = []

    if rulesets is None:
    
        for i, ant in enumerate(colony['ants']):
            if len(ant['rule']) == 0 or ant['rule'][-1][1] != positive_class:
                continue
    
            dominated = False
            for solution in archive:
                if dominates(solution["fitness"], ant["fitness"]):
                    dominated = True
                    break
    
            if not dominated:
                # Remove all solutions dominated by current ant solution
                archive = [solution for solution in archive if not dominates(ant["fitness"], solution["fitness"])]
    
                # Add the new non-dominated solution
                if not any(set(solution["rule"]) == set(ant["rule"]) for solution in archive):
                    archive.append(ant)
                    best_ants_indices.append(i)

    elif rulesets == 'iteration':

        sorted_colony = sorted(colony['ants'], key=lambda ant: ant['f1_score'], reverse=True)
        ruleset = []

        for i, ant in enumerate(sorted_colony):
            if len(ant['rule']) == 0 or ant['rule'][-1][1] != positive_class:
                continue

            # check duplicates in the ruleset
            if any(set(rule["rule"]) == set(ant["rule"]) for rule in ruleset):
                continue

            dominated = False
            for solution in archive:
                for rule in solution["ruleset"]:
                    if dominates(rule["fitness"], ant["fitness"]):
                        dominated = True
                        break

            if not dominated and len(ruleset) < ruleset_size:
                # Add the new non-dominated solution to ruleset
                ruleset.append(ant)
                best_ants_indices.append(i)

        # Add the ruleset to the archive
        if ruleset:
            archive.append({
                "ruleset": ruleset,
                "f1_score": sum(ant["f1_score"] for ant in ruleset) / len(ruleset),
            })
        else:
            best_ants_indices = []

    elif rulesets == 'subproblem':
        
        if not archive:
            archive = [{'f1_score': 0.0, 'ruleset': []} for _ in range(len(colony['ants']))]

        for i, ant in enumerate(colony['ants']):
            if len(ant['rule']) == 0 or ant['rule'][-1][1] != positive_class:
                continue

            ruleset = archive[i].get("ruleset", [])

            # check duplicates in the ruleset
            if any(set(rule["rule"]) == set(ant["rule"]) for rule in ruleset):
                continue

            if len(ruleset) < ruleset_size:
                dominated = False
                for rule in ruleset:
                    if dominates(rule["fitness"], ant["fitness"]):
                        dominated = True
                        break

                ruleset.append(ant)
                if not dominated:    
                    best_ants_indices.append(i)
            else:
                # Check if the new ant is better than the worst in the ruleset
                worst_rule = min(ruleset, key=lambda rule: rule["f1_score"])
                if dominates(ant["fitness"], worst_rule["fitness"]):
                    # Replace the worst rule with the new ant
                    ruleset.remove(worst_rule)
                    ruleset.append(ant)
                    best_ants_indices.append(i)

            if ruleset:
                archive[i]["ruleset"] = ruleset
                archive[i]["f1_score"] = sum(rule["f1_score"] for rule in ruleset) / len(ruleset) if ruleset else 0
        
    else:
        raise ValueError("Invalid rulesets type. Expected 'iteration', 'subproblem', or None.")

    return best_ants_indices, archive