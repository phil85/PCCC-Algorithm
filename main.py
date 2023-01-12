# %% Import packages
import pandas as pd
from pccc import pccc
from sklearn.metrics import adjusted_rand_score

# %% Generate data for illustrative example

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

df = pd.DataFrame(data, columns=['x1', 'x2', 'class'])

#%% Define problem input parameters and sets

n_clusters = df['class'].unique().size
X = df[['x1', 'x2']].values
y = df['class'].values

# %% Define information on pairs of objects

ml = [(2, 6), (9, 10)]
cl = [(2, 7), (6, 7), (5, 11)]
sml = [(1, 2), (1, 6)]
scl = [(4, 8), (8, 13)]
sml_weights = [0.8, 0.6]
scl_weights = [0.9, 0.5]

# %% Apply algorithm
labels = pccc(X, n_clusters, ml=ml, cl=cl, sml=sml, scl=scl, sml_weights=sml_weights, scl_weights=scl_weights,
              random_state=24)

# %% Evaluate assignment
print('ARI: ', adjusted_rand_score(y, labels))
