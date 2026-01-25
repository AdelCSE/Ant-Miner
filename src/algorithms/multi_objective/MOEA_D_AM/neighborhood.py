from .decomposition import get_scalarized
from .fitness import fitness_function

def find_best_neighborhood_rule(colony, data, ant_index, neighborhood, rep_list, neighbors, weights, decomposition, reference, labels, task, objs):

    # Compute current ant's fitness scalar value
    if task == 'single':
        best_ruleset = [colony['ants'][ant_index]['rule']]
        fitness = colony['ants'][ant_index]['fitness']
    else:
        best_ruleset = [rule['rule'] for rule in colony['ants'][ant_index]['ruleset']['rules'] if len(rule['rule']) > 0]
        fitness = colony['ants'][ant_index]['ruleset']['fitness']

    ant_g = get_scalarized(fitness=fitness, reference=reference, weights=weights[ant_index], approach=decomposition)

    changed_rule = False

    for i in neighborhood[ant_index][:neighbors]:

        # Skip if the ant is itself
        if i == ant_index:
            continue

        if task == 'single':
            new_ruleset = [colony['ants'][i]['rule']]
            fitness = colony['ants'][i]['fitness']
        else:
            new_ruleset = [{'rule': rule['rule'], 'scores': rule['scores']} for rule in colony['ants'][i]['ruleset']['rules'] if len(rule['rule']) > 0]
            fitness = colony['ants'][i]['ruleset']['fitness']

        if len(new_ruleset) == 0 or len(new_ruleset[0]) == 0:
            continue

        new_g = get_scalarized(fitness=fitness, reference=reference, weights=weights[i], approach=decomposition)

        # Check if new_rule is already in the replacing solution list
        in_rep_list = False
        for past_ruleset in rep_list[ant_index]:
             if set(map(tuple, new_ruleset)) == set(map(tuple, past_ruleset)):
                in_rep_list = True
                break
             
        if in_rep_list:
            continue

        if task == 'single':
            
            # If neighbor solution better, replace current
            if new_g > ant_g and colony['ants'][i]['rule'][-1][1] == 'pos':
                colony['ants'][ant_index]['rule'] = new_ruleset[0]
                colony['ants'][ant_index]['fitness'], colony['ants'][ant_index]['score'] = fitness_function(data=data, ant=colony['ants'][ant_index], labels=['class'], task=task, objs=objs)
                ant_g = new_g
                best_ruleset = new_ruleset
                changed_rule = True
        else:
            # If neighbor solution better, replace current
            if new_g > ant_g:
                colony['ants'][ant_index]['ruleset']['rules'] = [{'rule': rule['rule'], 'scores': rule['scores']} for rule in new_ruleset]
                colony['ants'][ant_index]['ruleset']['fitness'], colony['ants'][ant_index]['ruleset']['score'] = fitness_function(data=data, ant=colony['ants'][ant_index], labels=labels, task=task, objs=objs)
                ant_g = new_g
                best_ruleset = new_ruleset
                changed_rule = True

    # If tour changed, record it to avoid future reuse
    if changed_rule:
        rep_list[ant_index].append(best_ruleset)

    return colony, rep_list