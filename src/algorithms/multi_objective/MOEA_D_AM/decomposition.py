def get_scalarized(fitness : list[float], 
                  reference : list[float], 
                  weights : list[float], 
                  approach : str = "weighted"
                  ) -> float:
    
    if approach == "weighted":
        return sum(weights[l] * fitness[l] for l in range(len(fitness)))
    elif approach == "tchebycheff":
        return max(weights[j] * abs(fitness[j] - reference[j]) for j in range(len(fitness)))
    else:
        raise ValueError(f"Unknown decomposition approach: {approach}! Use 'weighted' or 'tchebycheff'.")