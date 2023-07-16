class Person():
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print("Class initialized.")
    
    def set_name(self, name):
        self.name = name
        self.print_new_name()
    
    def get_name(self):
        return self.name

    def print_new_name(self):
        print(f"New name set: {self.name}")

def multiply_function(number, multiplier=2):
    result = number * multiplier
    return result, multiplier

numeric_variable = 123

text_variable = "456"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_heatmap(searcher, n_values, p_values, t_delta):

    results = pd.DataFrame.from_dict(searcher.cv_results_)
    
    results["params_str"] = results.params.apply(str)
    
    scores_matrix = results.sort_values("iter").pivot_table(
        index="param_n_neighbors",
        columns="param_p",
        values="mean_test_score",
        aggfunc="last",
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.grid(False)
    
    im = ax.imshow(scores_matrix, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(len(p_values)))
    ax.set_xticklabels([str(x) for x in p_values])
    ax.set_xlabel("Minkowski Distance", fontsize=14)

    ax.set_yticks(np.arange(len(n_values)))
    ax.set_yticklabels([str(x) for x in n_values])
    ax.set_ylabel("Neighbors", fontsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    iterations = results.pivot_table(
        index="param_n_neighbors", columns="param_p", values="iter", aggfunc="max"
    ).values
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            plt.text(
                j,
                i,
                iterations[i, j],
                ha="center",
                va="center",
                color="w",
                fontsize=160,
            )
    
    # Color bar:
    fig.subplots_adjust(right=1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("mean_test_score", rotation=-90, va="bottom", fontsize=14)
    
    ax.set_title(f"Successive Halving\ntime = {t_delta:.2f} sec", fontsize=14)
    plt.show()
