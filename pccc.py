import gurobipy as gb
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
import networkx as nx
import pandas as pd
import time
import warnings


def choose_initial_cluster_centers(data, n_clusters, **kwargs):
    # Get parameters
    n_datapoints = data.shape[0]
    init = kwargs.get('init', 'k-means++')
    seed = kwargs.get('random_state', 1)

    if init == 'k-means++':
        # Apply k-means++ algorithm
        random_state = check_random_state(seed)
        x_squared_norms = row_norms(data, squared=True)
        centers, _ = kmeans_plusplus(data, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms)

    else:
        # Set random seed
        np.random.seed(seed)

        # Select datapoints to serve as cluster centers
        center_ids = np.random.choice(np.arange(n_datapoints), size=n_clusters, replace=False)

        # Get feature values of these datapoints
        centers = data[center_ids, :]

    return centers


def reposition_cluster_centers(data, n_clusters, labels, current_centers, weights, scl, confidence,
                               cluster_repositioning):
    if cluster_repositioning == 'inertia':

        # Get inertia per cluster
        inertia_per_cluster = get_inertia_per_cluster(n_clusters, data, current_centers, labels, weights)

        # Rank clusters
        rank = np.argsort(inertia_per_cluster)

        # Reposition top cluster
        current_centers[rank[0], :] = current_centers[rank[-1], :]

    elif cluster_repositioning == 'violations':

        # Get weighted violations per cluster
        weighted_violations_per_cluster = get_weighted_violations_per_cluster(n_clusters, labels, scl, confidence)

        # Rank clusters
        rank = np.argsort(weighted_violations_per_cluster)

        # Reposition top cluster
        current_centers[rank[0], :] = current_centers[rank[-1], :]

    elif cluster_repositioning == 'violations_inertia':

        # Get weighted violations per cluster
        weighted_violations_per_cluster = get_weighted_violations_per_cluster(n_clusters, labels, scl, confidence)

        # Get inertia per cluster
        inertia_per_cluster = get_inertia_per_cluster(n_clusters, data, current_centers, labels, weights)

        # Rank clusters (first by violations, then by inertia)
        rank = np.lexsort((inertia_per_cluster, weighted_violations_per_cluster))

        # Reposition top cluster
        current_centers[rank[0], :] = current_centers[rank[-1], :]

    else:
        raise ValueError('Invalid repositioning_ranking_strategy')

    return current_centers


def get_weighted_violations_per_cluster(n_clusters, labels, scl, confidence):
    # Initialize violations per cluster
    weighted_violations_per_cluster = np.zeros(n_clusters)

    if len(scl) == 0:
        return weighted_violations_per_cluster

    # Identify violated constraints
    violated_scl_constraints = labels[scl[:, 0]] == labels[scl[:, 1]]

    # Get confidence of violated constraints
    confidence_of_violated_scl_constraints = confidence.loc[scl[violated_scl_constraints, :].tolist()]

    df_violated_scl_constraints = pd.DataFrame({'label_i': labels[scl[violated_scl_constraints, 0]],
                                                'label_j': labels[scl[violated_scl_constraints, 1]],
                                                'confidence': confidence_of_violated_scl_constraints.values})

    conf_label_i = df_violated_scl_constraints.groupby('label_i')['confidence'].sum()
    conf_label_j = df_violated_scl_constraints.groupby('label_j')['confidence'].sum()

    # Compute weighted violations per cluster
    for i in range(n_clusters):
        weighted_violations_per_cluster[i] = conf_label_i.get(i, 0) + conf_label_j.get(i, 0)

    return weighted_violations_per_cluster


def get_inertia_per_cluster(n_clusters, data, centers, labels, weights):
    # Compute inertia for each cluster
    inertia_per_object = ((data - centers[labels, :]) ** 2).sum(axis=1)

    inertia_per_cluster = np.zeros(n_clusters)
    for i in range(n_clusters):
        inertia_per_cluster[i] = (inertia_per_object[labels == i] * weights[labels == i]).sum()

    return inertia_per_cluster


def update_centers(data, centers, n_clusters, labels, weights, scl, confidence):
    # Identify empty clusters
    non_empty_clusters = np.unique(labels)
    empty_clusters = np.setdiff1d(np.arange(n_clusters), non_empty_clusters)

    # Check if there are empty clusters
    if len(empty_clusters) > 0:

        # Get weighted violations per cluster
        weighted_violations_per_cluster = get_weighted_violations_per_cluster(n_clusters, labels, scl, confidence)

        # Get inertia per cluster
        inertia_per_cluster = get_inertia_per_cluster(n_clusters, data, centers, labels, weights)

        # Rank clusters (first by weighted violations, then by inertia)
        rank = np.lexsort((inertia_per_cluster, weighted_violations_per_cluster))

        # Reposition empty clusters
        position = -1
        for i in empty_clusters:
            # Get object from last cluster
            candidates = np.where(labels == rank[position])[0]

            # Randomly select an object
            selected_candidate = np.random.choice(candidates)
            centers[i, :] = data[selected_candidate, :]

            # Assign object to cluster
            labels[selected_candidate] = i

            # Decrement position
            position -= 1

    # Update cluster positions
    for i in range(n_clusters):
        centers[i] = np.average(data[labels == i, :], axis=0, weights=weights[labels == i])

    return centers


def preprocessing(data, ml, cl, sml, scl, sml_weights, scl_weights):
    # If there are no hard must-link constraints most of the preprocessing is skipped
    if len(ml) == 0:
        weights = np.ones(data.shape[0])
        mapping = np.arange(data.shape[0])
        confidence, sml, scl = aggregate_confidence_values(scl, scl_weights, sml, sml_weights)
        kdt = KDTree(data)
        return data, weights, mapping, cl, sml, scl, confidence, kdt

    # Initialize objects
    n_objects = data.shape[0]
    weights = np.ones(n_objects, dtype=int)
    mapping = np.arange(n_objects)

    # Construct undirected graph based on hard must-link constraints
    must_link_graph = nx.from_edgelist(list(ml))

    # Get connected components
    connected_components = nx.connected_components(must_link_graph)

    # Contract nodes in each connected component and adjust cannot-link constraints
    for component in connected_components:
        # Get objects in connected component
        ids = np.array(list(component))

        # Use object with maximum index as representative
        max_id = max(ids)

        # Update mapping, weights, and positions of representatives
        mapping[ids] = max_id
        weights[max_id] = len(ids)
        data[max_id, :] = data[ids, :].mean(axis=0)

    # Get all representatives
    representatives = np.unique(mapping)

    # Only keep coordinates of representatives
    data = data[representatives, :]

    # Only keep weights of representatives
    weights = weights[representatives]

    # Label representatives from 0 to number of representatives
    mapping = pd.Categorical(mapping).codes

    # Remove redundant hard cannot-link constraints
    if len(cl) > 0:
        # Map cannot-link constraints to representatives
        cl_array = mapping[cl.ravel()].reshape(-1, 2)
        cl = drop_duplicate_hard_constraints(cl_array)

    # Aggregate soft constraints
    if len(scl) > 0:
        scl_array = mapping[scl.ravel()].reshape(-1, 2)
        scl, scl_weights = aggregate_soft_constraints(scl_array, scl_weights)
    if len(sml) > 0:
        sml_array = mapping[sml.ravel()].reshape(-1, 2)
        sml, sml_weights = aggregate_soft_constraints(sml_array, sml_weights)

    # Merge all weights associated with soft constraints into one pandas series named confidence
    confidence, sml, scl = aggregate_confidence_values(scl, scl_weights, sml, sml_weights)

    return data, weights, mapping, cl, sml, scl, confidence


def aggregate_confidence_values(scl, scl_weights, sml, sml_weights):
    # Return None if there are no soft constraints
    if (len(scl) == 0) and (len(sml) == 0):
        return None, sml, scl

    # Add soft cannot-link constraints
    if len(scl) > 0:
        # Add weights as negative values to array
        all_constraints = np.concatenate((scl, -1 * scl_weights.reshape(-1, 1)), axis=1)

    # Add soft must-link constraints
    if len(sml) > 0:
        if len(scl) > 0:
            sml_constraints = np.concatenate((sml, sml_weights.reshape(-1, 1)), axis=1)
            all_constraints = np.concatenate((all_constraints, sml_constraints), axis=0)
        else:
            all_constraints = np.concatenate((sml, sml_weights.reshape(-1, 1)), axis=1)

    # Create dataframe from array
    df = pd.DataFrame(all_constraints, columns=['i', 'j', 'confidence'])

    # Convert indices to integers
    df[['i', 'j']] = df[['i', 'j']].astype(int)

    # Aggregate confidence values
    confidence = df.groupby(['i', 'j'])['confidence'].sum()

    # Keep only relevant constraints
    constraints = confidence.reset_index().values
    idx_relevant = constraints[:, 0] != constraints[:, 1]
    constraints = constraints[idx_relevant, :]
    idx_scl = constraints[:, 2] < 0
    idx_sml = constraints[:, 2] > 0
    scl = constraints[idx_scl, :2].astype(int)
    sml = constraints[idx_sml, :2].astype(int)

    # Get absolute values
    confidence = confidence.abs()

    return confidence, sml, scl


def my_callback(model, where):
    if where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        best = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
        if elapsed_time > model._assignment_time_limit and best < gb.GRB.INFINITY:
            model.terminate()


def get_relevant_constraints_dynamic(constraints, n_clusters, n_neighbors, df_nearest_centers_unselected,
                                     df_nearest_centers_selected, n_neighbors_selected, selected_idx, unselected_idx):
    # Get number of cannot-link constraints
    n_constraints = constraints.shape[0]
    constraint_ids = np.arange(n_constraints)

    # Get nearest centers for unselected representatives in first column of matrix constraints
    constraints_unselected_idx = unselected_idx[constraints[:, 0]]
    nearest_centers_matrix_i_unselected = df_nearest_centers_unselected.loc[
        constraints[constraints_unselected_idx, 0]].values

    # Convert nearest centers matrix in sparse binary matrix
    row1 = np.repeat(constraint_ids[constraints_unselected_idx], n_neighbors)
    col1 = nearest_centers_matrix_i_unselected.ravel()
    val1 = np.ones(len(row1), dtype=bool)

    constraints_selected_idx = selected_idx[constraints[:, 0]]
    nearest_centers_matrix_i_selected = df_nearest_centers_selected.loc[
        constraints[constraints_selected_idx, 0]].values

    row2 = np.repeat(constraint_ids[constraints_selected_idx], n_neighbors_selected)
    col2 = nearest_centers_matrix_i_selected.ravel()
    val2 = np.ones(len(row2), dtype=bool)

    row = np.concatenate((row1, row2))
    col = np.concatenate((col1, col2))
    val = np.concatenate((val1, val2))

    nearest_centers_binary_matrix_i = csr_matrix((val, (row, col)), shape=(n_constraints, n_clusters), dtype=bool)

    # Get nearest centers for all representatives in second column of matrix constraints
    constraints_unselected_idx = unselected_idx[constraints[:, 1]]
    nearest_centers_matrix_j_unselected = df_nearest_centers_unselected.loc[
        constraints[constraints_unselected_idx, 1]].values

    # Convert nearest centers matrix in sparse binary matrix
    row1 = np.repeat(constraint_ids[constraints_unselected_idx], n_neighbors)
    col1 = nearest_centers_matrix_j_unselected.ravel()
    val1 = np.ones(len(row1), dtype=bool)

    constraints_selected_idx = selected_idx[constraints[:, 1]]
    nearest_centers_matrix_j_selected = df_nearest_centers_selected.loc[
        constraints[constraints_selected_idx, 1]].values

    row2 = np.repeat(constraint_ids[constraints_selected_idx], n_neighbors_selected)
    col2 = nearest_centers_matrix_j_selected.ravel()
    val2 = np.ones(len(row2), dtype=bool)

    row = np.concatenate((row1, row2))
    col = np.concatenate((col1, col2))
    val = np.concatenate((val1, val2))

    nearest_centers_binary_matrix_j = csr_matrix((val, (row, col)), shape=(n_constraints, n_clusters), dtype=bool)

    # Perform elementwise multiplication of both binary matrices to get intersections
    intersection_matrix = nearest_centers_binary_matrix_i.multiply(nearest_centers_binary_matrix_j)
    idx_constraints, idx_clusters = intersection_matrix.nonzero()
    relevant_constraints = intersection_matrix.max(axis=1).nonzero()[0]

    return idx_constraints, idx_clusters, relevant_constraints


def assign_objects(data, centers, weights, ml, cl, sml, scl, confidence, log, labels=None, **kwargs):
    # Get parameters
    n_representatives = data.shape[0]
    n_clusters = centers.shape[0]
    representatives = np.arange(n_representatives)

    # Get control parameters
    metric = kwargs.get('metric', 'squared_euclidean')
    penalty = kwargs.get('penalty', 'avg_distance')
    n_neighbors = kwargs.get('n_neighbors', n_clusters)
    time_limit = kwargs.get('time_limit', 1e6)
    assignment_time_limit = kwargs.get('assignment_time_limit', time_limit)
    verbose = kwargs.get('verbose', 0)
    dynamic_n_neighbors = kwargs.get('dynamic_n_neighbors', None)
    if dynamic_n_neighbors != 'none' and dynamic_n_neighbors is not None and labels is not None and len(scl) > 0:
        sorting_strategy = dynamic_n_neighbors.split('.')[0]
        selection_strategy = dynamic_n_neighbors.split('.')[1]
        increase_strategy = dynamic_n_neighbors.split('.')[2]
        timing_strategy = dynamic_n_neighbors.split('.')[3]
        cluster_repositioning_completed_flag = kwargs.get('cluster_repositioning_completed_flag', False)
        if timing_strategy == 'after_repositioning' and not cluster_repositioning_completed_flag:
            dynamic_n_neighbors = None
    else:
        dynamic_n_neighbors = None

    log_flag = kwargs.get('log_flag', False)

    # Create iteration log
    iteration_log = {}

    if log_flag:
        tic_setup = time.perf_counter()

    # Determine nearest cluster centers and distances
    selected = []
    if dynamic_n_neighbors is not None:

        # Identify violated constraints
        violated_scl_constraints = labels[scl[:, 0]] == labels[scl[:, 1]]

        if np.any(violated_scl_constraints):

            # Get confidence of violated constraints
            confidence_of_violated_scl_constraints = confidence.loc[scl[violated_scl_constraints, :].tolist()]

            df_violated_scl_constraints = pd.DataFrame({'i': scl[violated_scl_constraints, 0],
                                                        'j': scl[violated_scl_constraints, 1],
                                                        'confidence': confidence_of_violated_scl_constraints.values})

            conf_i = df_violated_scl_constraints.groupby('i')['confidence'].sum()
            conf_j = df_violated_scl_constraints.groupby('j')['confidence'].sum()

            representatives_with_violations, counts = np.unique(scl[violated_scl_constraints], return_counts=True)
            representatives_with_weighted_violations = pd.Series(0, index=representatives, dtype=np.int64)
            representatives_with_weighted_violations.loc[conf_i.index] = conf_i
            representatives_with_weighted_violations.loc[conf_j.index] += conf_j

            # Initialize violations per representative
            violations = pd.Series(0, index=representatives, dtype=np.int64)

            # Update violations
            violations.loc[representatives_with_violations] = counts

            if sorting_strategy == 'n_violations':

                # Sort violations in descending order
                sorted_representatives = representatives_with_weighted_violations.sort_values(ascending=False).index

            elif sorting_strategy == 'n_violations_neighbors':

                # Sort violations in descending order
                sorted_representatives = representatives_with_weighted_violations[
                    representatives_with_violations].sort_values(ascending=False).index

                # Get neighbors of representatives with violations
                representatives_with_violations_idx = np.zeros(n_representatives, dtype=bool)
                representatives_with_violations_idx[representatives_with_violations] = True
                neighbors_idx = np.zeros(n_representatives, dtype=bool)
                neighbors1_idx = representatives_with_violations_idx[scl[:, 0]]
                neighbors2_idx = representatives_with_violations_idx[scl[:, 1]]
                neighbors_idx[scl[neighbors1_idx, 1]] = True
                neighbors_idx[scl[neighbors2_idx, 0]] = True
                neighbors_idx = neighbors_idx & ~representatives_with_violations_idx

                # Add direct neighbors of representatives with a violation
                sorted_representatives = np.concatenate((sorted_representatives, representatives[neighbors_idx]))

                # Add remaining representatives
                remaining_representatives = np.setdiff1d(representatives, sorted_representatives)
                sorted_representatives = np.concatenate((sorted_representatives, remaining_representatives))

            elif sorting_strategy == 'within_cluster_violations':

                # Determine number of violations for each cluster
                violations_per_cluster = np.zeros(n_clusters, dtype=int)
                clusters_with_violations, counts = np.unique(labels[scl[violated_scl_constraints]], return_counts=True)
                for i, count in zip(clusters_with_violations, counts):
                    violations_per_cluster[i] = count

                # Sort clusters in descending order of number of violations
                rank = np.argsort(violations_per_cluster)[::-1]

                # Add members of clusters to list
                sorted_representatives = np.array([], dtype=int)
                for i in rank:
                    sorted_representatives = np.concatenate((sorted_representatives, representatives[labels == i]))

            else:
                raise ValueError('Invalid sorting strategy')

            if selection_strategy == 'all':
                n_selected = len(representatives_with_violations)
            elif selection_strategy.isnumeric():
                n_selected = int(selection_strategy)
            else:
                raise ValueError('Invalid selection strategy')

            # Adjust n_selected
            n_selected = min(n_selected, n_representatives - 1)

            # Determine selected and unselected representatives
            selected = sorted_representatives[:n_selected]
            unselected = np.setdiff1d(representatives, selected)
            selected_idx = np.zeros(n_representatives, dtype=bool)
            selected_idx[selected] = True
            unselected_idx = ~selected_idx

            # Compute distances to nearest cluster centers for unselected representatives
            kdtree = KDTree(centers)
            distances_unselected, nearest_centers_unselected = kdtree.query(data[unselected], k=n_neighbors)

            # Adjust distances
            if metric == 'squared_euclidean':
                distances_unselected = distances_unselected ** 2

            # Create dataframes
            df_distances_unselected = pd.DataFrame(distances_unselected, index=unselected)
            df_nearest_centers_unselected = pd.DataFrame(nearest_centers_unselected, index=unselected)

            if increase_strategy == 'n_clusters':
                n_neighbors_selected = n_clusters
            elif increase_strategy.isnumeric():
                n_neighbors_selected = min(int(increase_strategy), n_clusters)
            else:
                raise ValueError('Invalid increase strategy')

            if verbose >= 2:
                print('n_representatives_with_violations:', len(representatives_with_violations))
                print('n_selected:', len(selected))
                print('n_unselected:', len(unselected))
                print('total_n_violations:', violations.sum())
                print('max_violations_among_selected:', violations.loc[selected].max())
                print('mean_violations_among_selected:', violations.loc[selected].mean())
                print('min_violations_among_selected:', violations.loc[selected].min())
                print('n_neighbors_selected:', n_neighbors_selected)

            distances_selected, nearest_centers_selected = kdtree.query(data[selected], k=n_neighbors_selected)

            # Adjust distances
            if metric == 'squared_euclidean':
                distances_selected = distances_selected ** 2

            # Create dataframes
            df_distances_selected = pd.DataFrame(distances_selected, index=selected)
            df_nearest_centers_selected = pd.DataFrame(nearest_centers_selected, index=selected)

            distances = np.concatenate((distances_selected[:, :n_neighbors].ravel(),
                                        distances_unselected[:, :n_neighbors].ravel()))
        else:
            kdtree = KDTree(centers)
            distances, nearest_centers = kdtree.query(data, k=n_neighbors)

            # Adjust distances
            if metric == 'squared_euclidean':
                distances = distances ** 2
    else:
        kdtree = KDTree(centers)
        distances, nearest_centers = kdtree.query(data, k=n_neighbors)

        # Adjust distances
        if metric == 'squared_euclidean':
            distances = distances ** 2

    # Determine penalty for violating a soft constraint with confidence 1
    if penalty == 'auto':
        P = distances.max() * weights.max() + 1
    elif penalty == 'max_distance':
        P = distances.max()
    elif penalty == 'avg_distance':
        P = distances.mean()
    elif penalty == 'quartile_distance':
        P = np.quantile(distances, 0.25)
    elif penalty == 'third_quartile_distance':
        P = np.quantile(distances, 0.75)
    else:
        P = penalty

    # Create model
    m = gb.Model()

    # Create dictionary which contains decision variables
    if dynamic_n_neighbors is not None and len(selected) > 0:

        # Add variables from unselected representatives
        distances_unselected = {(i, df_nearest_centers_unselected.at[i, j]):
                                df_distances_unselected.at[i, j] * weights[i]
                                for i in unselected for j in range(n_neighbors)}

        # Add variables from selected representatives
        distances_selected = {(i, df_nearest_centers_selected.at[i, j]):
                              df_distances_selected.at[i, j] * weights[i]
                              for i in selected for j in range(n_neighbors_selected)}

        distances = {**distances_unselected, **distances_selected}
    else:
        distances = {(i, nearest_centers[i, j]): distances[i, j] * weights[i]
                     for i in range(n_representatives) for j in range(n_neighbors)}

    # Add decision variables to model
    x = m.addVars(distances.keys(), vtype=gb.GRB.BINARY, obj=distances)

    # Provide warm start
    if labels is not None:
        for i in range(n_representatives):
            if (i, labels[i]) in x.keys():
                x[i, labels[i]].Start = 1

    # Add hard cannot-link constraints
    if len(cl) > 0:
        # Get relevant constraints
        idx_constraints, idx_clusters, relevant_constraints = \
            get_relevant_constraints(cl, n_clusters, n_neighbors, nearest_centers)

        if log_flag:
            iteration_log['cl'] = len(relevant_constraints)

        # Add hard cannot-link constraints
        m.addConstrs(x[i, j] + x[i_, j] <= 1 for i, i_, j in zip(cl[idx_constraints, 0], cl[idx_constraints, 1],
                                                                 idx_clusters))

    # Add soft cannot-link constraints
    if len(scl) > 0:
        # Get relevant constraints
        if dynamic_n_neighbors is not None and len(selected) > 0:
            idx_constraints, idx_clusters, relevant_constraints = \
                get_relevant_constraints_dynamic(scl, n_clusters, n_neighbors, df_nearest_centers_unselected,
                                                 df_nearest_centers_selected, n_neighbors_selected, selected_idx,
                                                 unselected_idx)
        else:
            idx_constraints, idx_clusters, relevant_constraints = \
                get_relevant_constraints(scl, n_clusters, n_neighbors, nearest_centers)

        if log_flag:
            iteration_log['scl'] = len(relevant_constraints)

        # Add slack variables for soft cannot-link constraints
        keys = list(zip(scl[relevant_constraints, 0], scl[relevant_constraints, 1]))
        values = confidence.loc[keys] * P
        scl_constraints = dict(zip(keys, values))
        z = m.addVars(scl_constraints.keys(), lb=0, ub=1, obj=scl_constraints, name='z')

        # Add soft cannot-link constraints
        m.addConstrs(x[i, j] + x[i_, j] <= 1 + z[i, i_] for i, i_, j in zip(scl[idx_constraints, 0],
                                                                            scl[idx_constraints, 1],
                                                                            idx_clusters))

    if len(ml) > 0 and weights.max() == 1:
        # Add hard must-link constraints
        m.addConstrs(x[i, j] == x[i_, j] for j in range(n_clusters) for i, i_ in ml)

    # Add soft must-link constraints
    if len(sml) > 0:

        # Get relevant constraints
        if dynamic_n_neighbors is not None and len(selected) > 0:
            idx_constraints, idx_clusters, relevant_constraints = \
                get_relevant_constraints_dynamic(sml, n_clusters, n_neighbors, df_nearest_centers_unselected,
                                                 df_nearest_centers_selected, n_neighbors_selected, selected_idx,
                                                 unselected_idx)
        else:
            idx_constraints, idx_clusters, relevant_constraints = \
                get_relevant_constraints(sml, n_clusters, n_neighbors, nearest_centers)

        if log_flag:
            iteration_log['sml'] = len(relevant_constraints)

        # Add slack variables for soft must-link constraints
        keys = list(zip(sml[relevant_constraints, 0], sml[relevant_constraints, 1]))
        values = confidence.loc[keys] * P
        sml_constraints = dict(zip(keys, values))
        w = m.addVars(sml_constraints.keys(), lb=0, ub=1, obj=sml_constraints, name='w')

        # Add soft must-link constraints
        m.addConstrs(x[i, j] - x[i_, j] <= w[i, i_] for i, i_, j in zip(sml[idx_constraints, 0],
                                                                        sml[idx_constraints, 1], idx_clusters))
        m.addConstrs(x[i_, j] - x[i, j] <= w[i, i_] for i, i_, j in zip(sml[idx_constraints, 0],
                                                                        sml[idx_constraints, 1], idx_clusters))

        # Add soft must-link constraints as soft constraints
        m.addConstrs(x[sml[c, 0], j] <= w[sml[c, 0], sml[c, 1]]
                     for c in relevant_constraints
                     for j in np.setdiff1d(nearest_centers[sml[c, 0], :], nearest_centers[sml[c, 1], :]))
        m.addConstrs(x[sml[c, 1], j] <= w[sml[c, 0], sml[c, 1]]
                     for c in relevant_constraints
                     for j in np.setdiff1d(nearest_centers[sml[c, 1], :], nearest_centers[sml[c, 0], :]))

    # Each representative must be assigned to a cluster
    m.addConstrs(x.sum(i, '*') == 1 for i in range(n_representatives))

    # Set solver parameters
    if verbose < 2:
        m.setParam('OutputFlag', 0)
    if assignment_time_limit is not None:
        m._assignment_time_limit = assignment_time_limit
    if log_flag:
        iteration_log['cpu_model_setup'] = time.perf_counter() - tic_setup
    if len(cl) == 0:
        m.setParam('TimeLimit', assignment_time_limit)
    else:
        m.setParam('TimeLimit', time_limit)

    m.setParam('MipFocus', 1)

    # Run optimization
    m.optimize(my_callback)

    # Check if feasible solution was found (continue optimization if total elapsed time does not exceed time limit)
    if m.SolCount == 0:
        return None, None, log

    # Get labels from optimal assignment
    labels = np.zeros(n_representatives, dtype=int)
    for i, j in x.keys():
        if x[i, j].X > 0.5:
            labels[i] = j

    # Get total penalty
    total_penalty = m.ObjVal - sum([v.Obj for v in x.values() if v.X > 0.5])

    if log_flag:
        iteration_log['penalty'] = P
        iteration_log['n_scl_vio_in_assignment'] = len([(i, j) for i, j in scl if labels[i] == labels[j]])
        iteration_log['n_sml_vio_in_assignment'] = len([(i, j) for i, j in sml if labels[i] != labels[j]])
        if len(scl) > 0:
            iteration_log['n_scl_vio_in_optimization'] = z.sum().getValue()
        else:
            iteration_log['n_scl_vio_in_optimization'] = 0
        if len(sml) > 0:
            iteration_log['n_sml_vio_in_optimization'] = w.sum().getValue()
        else:
            iteration_log['n_sml_vio_in_optimization'] = 0
        iteration_log['ofv_distance'] = m.ObjVal - total_penalty
        iteration_log['ofv_penalty'] = total_penalty
        iteration_log['ofv_total'] = m.ObjVal
        iteration_log['status'] = m.Status
        iteration_log['mipgap'] = m.MIPGap
        iteration_log['cpu_model_solve'] = m.Runtime
        log['iterations'].append(iteration_log)

    return labels, total_penalty, log


def get_relevant_constraints(constraints, n_clusters, n_neighbors, nearest_centers):
    # Get number of cannot-link constraints
    n_constraints = constraints.shape[0]

    # Get nearest centers for all representatives in first column of matrix constraints
    nearest_centers_matrix_i = nearest_centers[constraints[:, 0]]

    # Convert nearest centers matrix in sparse binary matrix
    row = np.repeat(range(n_constraints), n_neighbors)
    col = nearest_centers_matrix_i.ravel()
    val = np.ones(len(row), dtype=bool)
    nearest_centers_binary_matrix_i = csr_matrix((val, (row, col)), shape=(n_constraints, n_clusters), dtype=bool)

    # Get nearest centers for all representatives in second column of matrix constraints
    nearest_centers_matrix_j = nearest_centers[constraints[:, 1]]

    # Convert nearest centers matrix in sparse binary matrix
    row = np.repeat(range(n_constraints), n_neighbors)
    col = nearest_centers_matrix_j.ravel()
    val = np.ones(len(row), dtype=bool)
    nearest_centers_binary_matrix_j = csr_matrix((val, (row, col)), shape=(n_constraints, n_clusters), dtype=bool)

    # Perform elementwise multiplication of both binary matrices to get intersections
    intersection_matrix = nearest_centers_binary_matrix_i.multiply(nearest_centers_binary_matrix_j)
    idx_constraints, idx_clusters = intersection_matrix.nonzero()
    relevant_constraints = intersection_matrix.max(axis=1).nonzero()[0]

    return idx_constraints, idx_clusters, relevant_constraints


def get_total_distance(data, centers, labels, weights, **kwargs):
    metric = kwargs.get('metric', 'euclidean')

    if metric == 'euclidean':
        dist = (np.sqrt(((data - centers[labels, :]) ** 2).sum(axis=1)) * weights).sum()
    elif metric == 'squared_euclidean':
        dist = (((data - centers[labels, :]) ** 2).sum(axis=1) * weights).sum()
    else:
        raise ValueError('Invalid metric')

    return dist


def drop_duplicate_hard_constraints(constraints):
    if len(constraints) > 0:
        nodes_i = constraints.min(axis=1)
        nodes_j = constraints.max(axis=1)
        constraints = np.stack((nodes_i, nodes_j), axis=1)
        constraints = np.unique(constraints, axis=0)
    return constraints


def aggregate_soft_constraints(constraints, weights):
    if len(constraints) > 0:
        nodes_i = constraints.min(axis=1)
        nodes_j = constraints.max(axis=1)
        constraints = np.stack((nodes_i, nodes_j), axis=1)
        constraints, ind = np.unique(constraints, axis=0, return_inverse=True)
        weights = pd.Series(weights).groupby(ind).sum().values
    return constraints, weights


def check_input(ml, cl, sml, scl, sml_weights, scl_weights, n_clusters, **kwargs):
    # Initialize optional arguments
    n_neighbors = kwargs.get('n_neighbors', n_clusters)

    if ml is None:
        ml = []
    if cl is None:
        cl = []
    if sml is None:
        sml = []
    if scl is None:
        scl = []
    if sml_weights is None:
        if len(sml) > 0:
            sml_weights = np.ones(len(sml))
        else:
            sml_weights = []
    if scl_weights is None:
        if len(scl) > 0:
            scl_weights = np.ones(len(scl))
        else:
            scl_weights = []

    # Convert lists to arrays
    ml = np.array([*ml])
    cl = np.array([*cl])
    sml = np.array([*sml])
    scl = np.array([*scl])
    sml_weights = np.array(sml_weights)
    scl_weights = np.array(scl_weights)

    # Remove redundant hard cannot-link constraints
    cl = drop_duplicate_hard_constraints(cl)
    ml = drop_duplicate_hard_constraints(ml)

    # Aggregate soft constraints
    sml, sml_weights = aggregate_soft_constraints(sml, sml_weights)
    scl, scl_weights = aggregate_soft_constraints(scl, scl_weights)

    return ml, cl, sml, scl, sml_weights, scl_weights, n_neighbors


def adjust_n_neighbors(cl, n_clusters, n_neighbors):
    if len(cl) > 0:
        _, counts = np.unique(cl, return_counts=True)
        lower_bound = max(counts.max() + 1, n_neighbors)
        n_neighbors = min(lower_bound, n_clusters)
        return n_neighbors, counts.max()
    else:
        n_neighbors = min(n_neighbors, n_clusters)
    return n_neighbors, 0


def initialize_log(log, X, cl, ml, scl, sml):
    log['n_objects'] = X.shape[0]
    if ml is not None:
        log['ml'] = len(ml)
    else:
        log['ml'] = 0
    if cl is not None:
        log['cl'] = len(cl)
    else:
        log['cl'] = 0
    if sml is not None:
        log['sml'] = len(sml)
    else:
        log['sml'] = 0
    if scl is not None:
        log['scl'] = len(scl)
    else:
        log['scl'] = 0


def pccc(X, n_clusters, ml=None, cl=None, sml=None, scl=None, sml_weights=None, scl_weights=None, **kwargs):
    log_flag = kwargs.get('log_flag', False)
    max_iter = kwargs.get('max_iter', 1e6)
    time_limit = kwargs.get('time_limit', 1e6)
    perform_preprocessing = kwargs.get('perform_preprocessing', True)
    cluster_repositioning = kwargs.get('cluster_repositioning', None)
    if cluster_repositioning == 'none':
        cluster_repositioning = None
    if cluster_repositioning is None:
        kwargs['cluster_repositioning_completed_flag'] = True
    else:
        kwargs['cluster_repositioning_completed_flag'] = False
    results = {}

    # Start stopwatch
    tic = time.perf_counter()

    log = {}
    if log_flag:
        initialize_log(log, X, cl, ml, scl, sml)

    # Set initial parameters
    ml, cl, sml, scl, sml_weights, scl_weights, n_neighbors = check_input(ml, cl, sml, scl, sml_weights,
                                                                          scl_weights, n_clusters,
                                                                          **kwargs)

    # Copy data
    data = X.astype(float).copy()

    if log_flag:
        tic_preprocessing = time.perf_counter()

    # Perform preprocessing
    if perform_preprocessing:
        data, weights, mapping, cl, sml, scl, confidence = preprocessing(data, ml, cl, sml, scl, sml_weights,
                                                                         scl_weights)
    else:
        weights = np.ones(data.shape[0])
        mapping = np.arange(data.shape[0])
        confidence, sml, scl = aggregate_confidence_values(scl, scl_weights, sml, sml_weights)

    # Perform feasibility check
    if data.shape[0] < n_clusters:
        warnings.warn('There is no feasible assignment!')
        return results

    if log_flag:
        log['cpu_preprocessing'] = time.perf_counter() - tic_preprocessing
        log['n_ml_after_preprocessing'] = 0
        log['n_cl_after_preprocessing'] = cl.shape[0]
        log['n_sml_after_preprocessing'] = sml.shape[0]
        log['n_scl_after_preprocessing'] = scl.shape[0]
        log['n_representatives'] = data.shape[0]
        log['n_neighbors'] = n_neighbors
        log['iterations'] = []

    # Adjust n_neighbors based on hard cannot-link constraints
    n_neighbors, max_degree = adjust_n_neighbors(cl, n_clusters, n_neighbors)
    kwargs['n_neighbors'] = n_neighbors

    # Choose initial cluster centers randomly
    centers = choose_initial_cluster_centers(data, n_clusters, **kwargs)

    # Assign objects
    initial_labels = None
    labels, total_penalty, log = assign_objects(data, centers, weights, ml, cl, sml, scl, confidence, log,
                                                labels=initial_labels, **kwargs)

    # Perform feasibility check
    if labels is None:
        warnings.warn('No feasible assignment found!')
        return results

    # Initialize best labels
    best_labels = labels
    global_best_labels = best_labels

    # Update centers
    centers = update_centers(data, centers, n_clusters, labels, weights, scl, confidence)

    # Initialize best centers
    best_centers = centers.copy()
    global_best_centers = centers.copy()

    # Compute solution quality
    best_solution_quality = get_total_distance(data, centers, labels, weights, **kwargs) + total_penalty
    global_best_solution_quality = best_solution_quality

    n_iter = 0
    elapsed_time = time.perf_counter() - tic
    while (n_iter < max_iter) and (elapsed_time < time_limit):

        # Assign objects
        labels, total_penalty, log = assign_objects(data, centers, weights, ml, cl, sml, scl, confidence, log,
                                                    labels=labels, **kwargs)

        # Perform feasibility check
        if labels is None:
            warnings.warn('There is no feasible assignment!')
            return results

        # Update centers
        centers = update_centers(data, centers, n_clusters, labels, weights, scl, confidence)

        # Compute solution quality
        solution_quality = get_total_distance(data, centers, labels, weights, **kwargs) + total_penalty

        # Check stopping criterion
        if solution_quality >= best_solution_quality:

            if cluster_repositioning is not None and data.shape[0] > n_clusters:

                if best_solution_quality >= global_best_solution_quality:

                    if kwargs['cluster_repositioning_completed_flag'] is False:
                        labels = global_best_labels
                        centers = global_best_centers.copy()
                        kwargs['cluster_repositioning_completed_flag'] = True
                        best_solution_quality = 1e15
                    else:
                        break
                else:
                    # Update the global best labels and global best solution quality
                    global_best_labels = best_labels
                    global_best_centers = best_centers.copy()
                    global_best_solution_quality = best_solution_quality

                    # Reposition clusters
                    centers = reposition_cluster_centers(data, n_clusters, best_labels, best_centers, weights, scl,
                                                         confidence, cluster_repositioning)
                    labels = None

                    # Reinitialize best_solution_quality
                    best_solution_quality = 1e15

                    # Add information to performance log
                    if log_flag:
                        if 'cluster_repositionings' in log:
                            log['cluster_repositionings'] += 1
                        else:
                            log['cluster_repositionings'] = 1

                        if 'repositioning_iterations' in log:
                            log['repositioning_iterations'].append(n_iter)
                        else:
                            log['repositioning_iterations'] = [n_iter]
            else:
                break
        else:
            # Update best labels and best total distance
            best_labels = labels
            best_centers = centers.copy()
            best_solution_quality = solution_quality

        # Increase iteration counter and compute elapsed time
        n_iter += 1
        elapsed_time = time.perf_counter() - tic

    if log_flag:
        log['n_iter'] = n_iter + 2
        log['n_neighbors_after_adjustment'] = n_neighbors
        log['max_degree'] = max_degree
        log['total_runtime'] = time.perf_counter() - tic

    if global_best_solution_quality < best_solution_quality:
        best_labels = global_best_labels
        best_centers = global_best_centers.copy()

    results['labels'] = best_labels[mapping]
    results['data'] = data
    results['weights'] = weights
    results['cl'] = cl
    results['labels_representatives'] = best_labels
    results['centers_representatives'] = best_centers
    results['log'] = log

    return results
