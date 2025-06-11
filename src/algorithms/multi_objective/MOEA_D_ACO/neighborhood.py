from .pheromones import find_gmax
from .fitness import fitness_function

def find_best_neighborhood_rule(colony, data, ant_index, neighborhood, replacing_solution, T, lambda_weights):

    # Compute current ant's fitness scalar value
    ant_g = find_gmax(colony, lambda_weights, ant_index, ant_index)
    best_rule = colony['ant'][ant_index]['rule']
    changed_rule = False

    for i in neighborhood[ant_index][:T]:
        if i == ant_index:
            continue

        new_rule = colony['ant'][i]['rule']
        new_g = find_gmax(colony, lambda_weights, i, i)

        # Check if new_rule was already used to replace this ant
        in_replacing_solution = False
        for past_rule in replacing_solution[ant_index]:
            if new_rule == past_rule:
                in_replacing_solution = True
                break
        if in_replacing_solution:
            continue

        # If neighbor solution better, replace current
        if new_g > ant_g:
            colony['ant'][ant_index]['rule'] = new_rule
            colony['ant'][ant_index]['fitness'] = fitness_function(data=data, rule=new_rule)
            ant_g = new_g
            best_rule = new_rule
            changed_rule = True

    # If tour changed, record it to avoid future reuse
    if changed_rule:
        replacing_solution[ant_index].append(best_rule)

    return colony, replacing_solution