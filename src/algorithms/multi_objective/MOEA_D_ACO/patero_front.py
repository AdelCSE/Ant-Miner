import numpy as np
import matplotlib.pyplot as plt

def plot_patero_front(archive: dict) -> None:
    """
    Plot the Pareto front (non-dominated solutions) in 2D objective space.
        
    Args:
    Archive : dict
        The archive containing the non-dominated solutions with their fitness values.
    """

    if len(archive) == 0:
        raise ValueError("Archive is empty. No solutions to plot!")

    fitness_values = np.array([ant['fitness'] for ant in archive])

    plt.figure(figsize=(8, 6))
    plt.scatter(fitness_values[:, 0], fitness_values[:, 1], c='blue', marker='o', label='Non-dominated solutions')
    plt.title('Pareto Front of Non-dominated Solutions')
    plt.xlabel('sensitivity')
    plt.ylabel('specificity')
    plt.grid(True)
    plt.legend()
    plt.show()