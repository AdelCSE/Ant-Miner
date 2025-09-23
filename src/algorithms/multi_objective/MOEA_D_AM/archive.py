def dominates(fitness1: list, fitness2: list) -> bool:
    return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))


def is_duplicate_rule(rule, archive_rules):
    return any(set(rule) == set(existing['rule']) for existing in archive_rules)

def is_duplicate_ruleset(ruleset, archive_rulesets):
    ruleset_sets = [set(tuple(term) for term in rule['rule']) for rule in ruleset['rules']]
    for existing in archive_rulesets:
        existing_sets = [set(tuple(term) for term in rule['rule']) for rule in existing['ruleset']['rules']]
        if all(any(rs == es for es in existing_sets) for rs in ruleset_sets):
            return True
    return False


def avg_f1(ruleset):
    return sum(rule["f1_score"] for rule in ruleset) / len(ruleset) if ruleset else 0.0


def update_archive_rules(colony, archive):
    best_ants_indices = []
    for i, ant in enumerate(colony['ants']):
        if not ant['rule'] or ant['rule'][-1][1] != 'pos':
            continue
        if any(dominates(a['fitness'], ant['fitness']) for a in archive):
            continue
        archive = [a for a in archive if not dominates(ant['fitness'], a['fitness'])]
        if not is_duplicate_rule(ant['rule'], archive):
            archive.append(ant)
            best_ants_indices.append(i)
    return best_ants_indices, archive


def update_archive_iteration_rulesets(colony, archive, ruleset_size):
    best_ants_indices = []
    sorted_ants = sorted(colony['ants'], key=lambda a: a['f1_score'], reverse=True)
    ruleset = []

    for i, ant in enumerate(sorted_ants):
        if not ant['rule'] or ant['rule'][-1][1] != 'pos':
            continue
        if is_duplicate_rule(ant['rule'], ruleset):
            continue
        if any(dominates(rs['fitness'], ant['fitness']) for rs in archive):
            continue
        if len(ruleset) < ruleset_size:
            ruleset.append(ant)
            best_ants_indices.append(i)

    if ruleset:
        archive.append({"ruleset": ruleset, "f1_score": avg_f1(ruleset), "fitness": [sum(f) / len(f) for f in zip(*(r['fitness'] for r in ruleset))]})
    else:
        best_ants_indices = []

    return best_ants_indices, archive


def update_archive_subproblem_rulesets(colony, archive, ruleset_size):
    best_ants_indices = []

    if not archive:
        archive = [{'f1_score': 0.0, 'fitness': [0.0, 0.0], 'ruleset': []} for _ in range(len(colony['ants']))]

    for i, ant in enumerate(colony['ants']):
        if not ant['rule'] or ant['rule'][-1][1] != 'pos':
            continue

        ruleset = archive[i].get('ruleset', [])

        if is_duplicate_rule(ant['rule'], ruleset):
            continue

        if len(ruleset) < ruleset_size:
            if not any(dominates(r['fitness'], ant['fitness']) for r in ruleset):
                best_ants_indices.append(i)
            ruleset.append(ant)
        else:
            worst = min(ruleset, key=lambda r: r['f1_score'])
            if dominates(ant['fitness'], worst['fitness']):
                ruleset.remove(worst)
                ruleset.append(ant)
                best_ants_indices.append(i)

        archive[i]['ruleset'] = ruleset
        archive[i]['f1_score'] = avg_f1(ruleset)
        archive[i]['fitness'] = [sum(f) / len(f) for f in zip(*(r['fitness'] for r in ruleset))] if ruleset else [0.0, 0.0]

    return best_ants_indices, archive


def update_mlc_archive(colony: dict, archive: list) -> tuple:
    best_ants_indices = []

    for i, ant in enumerate(colony['ants']):
        if not ant['ruleset']:
            continue

        #print(f'Ant {i}: {ant}')
        if any(dominates(a['ruleset']['fitness'], ant['ruleset']['fitness']) for a in archive):
            continue

        archive = [a for a in archive if not dominates(ant['ruleset']['fitness'], a['ruleset']['fitness'])]

        if not is_duplicate_ruleset(ant['ruleset'], archive):
            archive.append(ant)
            best_ants_indices.append(i)
            
    return best_ants_indices, archive


def update_archive(colony: dict, archive: list, rulesets: str = None) -> tuple:
    """
    Updates the external Pareto archive with non-dominated solutions from the colony.

    Args:
        colony (dict): Colony containing ants with rules and fitness.
        archive (list): Existing archive of non-dominated solutions.
        rulesets (str): Strategy: None, 'iteration', or 'subproblem'.

    Returns:
        tuple: (list of contributing ant indices, updated archive)
    """
    if 'rule' in colony['ants'][0]:
        ruleset_size = 2
        if rulesets is None:
            return update_archive_rules(colony, archive)
        elif rulesets == 'iteration':
            return update_archive_iteration_rulesets(colony, archive, ruleset_size)
        elif rulesets == 'subproblem':
            return update_archive_subproblem_rulesets(colony, archive, ruleset_size)
        else:
            raise ValueError("Invalid rulesets type. Expected 'iteration', 'subproblem', or None.")
    
    else:
        return update_mlc_archive(colony, archive)
    

        
