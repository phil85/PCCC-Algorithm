# %% Import packages
import pandas as pd
from pccc import pccc
from sklearn.metrics import adjusted_rand_score

# %% Generate data for illustrative example from paper (https://arxiv.org/abs/2212.14437)

# Define data
data = [[0, 0, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 6, 1],
        [2, 0, 0],
        [2, 2, 0],
        [2, 4, 0],
        [2, 6, 1],
        [4, 0, 3],
        [4, 2, 2],
        [4, 4, 2],
        [4, 6, 2],
        [6, 0, 3],
        [6, 2, 2],
        [6, 4, 2],
        [6, 6, 2]]

df = pd.DataFrame(data, columns=['x1', 'x2', 'ground_truth_label'])

#%% Define problem input parameters and sets

n_clusters = df['ground_truth_label'].unique().size
X = df[['x1', 'x2']].values
y = df['ground_truth_label'].values

# %% Define information on pairs of objects

hard_mustlink_constraints = [(2, 6), (9, 10)]
hard_cannotlink_constraints = [(2, 7), (6, 7), (5, 11)]
soft_mustlink_constraints = [(1, 2), (1, 6)]
soft_cannotlink_constraints = [(4, 8), (8, 13)]
confidence_levels_of_soft_mustlink_constraints = [0.8, 0.6]
confidence_levels_of_soft_cannotlink_constraints = [0.9, 0.5]

# %% Apply algorithm
labels = pccc(X, n_clusters,
              ml=hard_mustlink_constraints,
              cl=hard_cannotlink_constraints,
              sml=soft_mustlink_constraints,
              scl=soft_cannotlink_constraints,
              sml_weights=confidence_levels_of_soft_mustlink_constraints,
              scl_weights=confidence_levels_of_soft_cannotlink_constraints,
              random_state=24)

# %% Evaluate assignment
print('ARI: ', adjusted_rand_score(y, labels))
