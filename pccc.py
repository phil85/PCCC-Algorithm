# © 2023, Universität Bern, Institut für Finanzmanagement, Philipp Baumann

import gurobipy as gb
from sklearn.neighbors import KDTree
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
import pandas as pd
import time
import warnings


def choose_initial_cluster_centers(data, n_clusters, seed, init):
    """

    Args:
        data (numpy.ndarray): datapoints
        n_clusters (int): number of clusters to be identified
        seed (int): random seed
        init (str): initialization method

    Returns:
        numpy.ndarray: positions of cluster centers
    """
    # Get parameters
    n_datapoints = data.shape[0]

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


def update_centers(data, centers, n_clusters, labels, weights):
    """

    Args:
        data (numpy.ndarray): feature vectors of objects
        centers (numpy.ndarray): current positions of cluster centers
        n_clusters (int): predefined number of clusters
        labels (numpy.ndarray): current cluster assignments of objects
        weights (numpy.ndarray): number of objects that are represented by representatives

    Returns:
        numpy.ndarray: the updated positions of cluster centers
    """
    for i in range(n_clusters):
        centers[i] = np.average(data[labels == i, :], axis=0, weights=weights[labels == i])
    return centers


def preprocessing(data, ml, cl, sml, scl, sml_weights, scl_weights):
    """

    Args:
        data (numpy.ndarray): feature vectors of objects
        ml (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        cl (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        sml (List[Tuple[int, int]]): tuples of objects that are  subject to a soft must-link constraint
        scl (List[Tuple[int, int]]): tuples of objects that are  subject to a soft cannot-link constraint
        sml_weights (List[float]): weights associated with soft must-link constraints
        scl_weights (List[float]): weights associated with soft cannot-link constraints

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        pandas.core.series.Series, sklearn.neighbors._kd_tree.KDTree]:

        data (np.array): feature vectors of representatives
        weights (np.array): number of objects that are represented by representatives
        mapping (np.array): maps objects (positions) to representatives (values)
        cl (np.array): array that contains hard cannot-link constraints on representatives as rows
        cl_weights (np.array): number of original cannot-link constraints that are represented by new cl constraints
        scl (np.array): array that contains soft cannot-link constraints on representatives as rows
        scl_weights (np.array): weights associated with soft cannot-link constraints that refer to representatives
        sml (np.array): array that contains soft must-link constraints on representatives as rows
        scl_weights (np.array): weights associated with soft must-link constraints that refer to representatives
        kdt (KDTree): kd-tree on objects (is used to find closest objects to empty clusters)
    """

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

    # Construct kdtree
    kdt = KDTree(data)

    return data, weights, mapping, cl, sml, scl, confidence, kdt


def aggregate_confidence_values(scl, scl_weights, sml, sml_weights):
    # Return None if there are no soft constraints
    if (len(scl) == 0) and (len(sml) == 0):
        return None, sml, scl
    else:
        all_constraints = None

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
    """ Callback to adjust time limit

    Args:
        model (gurobipy.Model): gurobi model
        where (int): where during the solution process the function is executed
    """
    if where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        best = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
        if elapsed_time > model._assignment_time_limit and best < gb.GRB.INFINITY:
            model.terminate()


def assign_objects(data, centers, weights, cl, sml, scl, confidence, kdt, control_parameters,
                   log_performance_statistics, performance_log, labels=None):
    """

    Args:
        data (numpy.ndarray): feature vectors of representatives
        centers (numpy.ndarray): current positions of cluster centers
        weights (numpy.ndarray): number of objects that are represented by representatives
        cl (numpy.ndarray): array that contains cannot-link constraints on representatives as rows
        sml (numpy.ndarray): array that contains soft must-link constraints on representatives as rows
        scl (numpy.ndarray): array that contains soft cannot-link constraints on representatives as rows
        confidence (pandas.core.series.Series): series that contains weights of soft constraints
        kdt (sklearn.neighbors._kd_tree.KDTree): kd-tree on objects (is used to find closest objects to empty clusters)
        control_parameters (Dict[str, str]): control parameters
        log_performance_statistics (bool): flag whether or not performance statistics are recorded
        performance_log (Dict[str, int]): dictionary that contains performance statistics
        labels (Union[None, numpy.ndarray]): current cluster labels of representatives

    Returns:
        Tuple[numpy.ndarray, float, Dict[str, int]]: labels, total_penalty, performance_log
    """

    # Get control parameters
    penalty = control_parameters['penalty']
    n_neighbors = control_parameters['n_neighbors']
    assignment_time_limit = control_parameters['assignment_time_limit']
    verbose = control_parameters['verbose']

    # Create iteration log
    iteration_log = {}

    if log_performance_statistics:
        tic_setup = time.perf_counter()

    # Get parameters
    n_representatives = data.shape[0]
    n_clusters = centers.shape[0]

    # Determine nearest cluster centers and distances
    kdtree = KDTree(centers)
    distances, nearest_centers = kdtree.query(data, k=n_neighbors)
    max_distance = distances.max()

    # Determine penalty for violating a soft constraint with confidence 1
    if penalty == 'max_distance':
        P = max_distance
    elif penalty == 'avg_distance':
        P = distances.mean()
    elif penalty == 'quartile_distance':
        P = np.quantile(distances, 0.25)
    else:
        P = penalty

    # Create model
    m = gb.Model()

    # Create dictionary which contains decision variables
    distances = {(i, nearest_centers[i, j]): distances[i, j] * weights[i]
                 for i in range(n_representatives) for j in range(n_neighbors)}

    # Include empty clusters
    empty_clusters = np.setdiff1d(range(n_clusters), np.unique(nearest_centers))

    if log_performance_statistics:
        iteration_log['empty_clusters'] = len(empty_clusters)

    if len(empty_clusters) > 0:
        # Determine closest representatives for each empty cluster
        distances_to_representatives, nearest_representatives = kdt.query(centers[empty_clusters, :])

        # Create dictionary
        keys = zip(nearest_representatives[:, 0], empty_clusters)
        values = distances_to_representatives[:, 0] * weights[nearest_representatives[:, 0]]
        distances_to_representatives = dict(zip(keys, values))

        # Merge dictionaries
        distances = {**distances, **distances_to_representatives}

    # Add decision variables to model
    x = m.addVars(distances.keys(), vtype=gb.GRB.BINARY, obj=distances)

    # Provide warm start
    if labels is not None:
        for i in range(len(labels)):
            if (i, labels[i]) in x.keys():
                x[i, labels[i]].Start = 1

    # Add hard cannot-link constraints
    if len(cl) > 0:
        # Get relevant constraints
        idx_constraints, idx_clusters, relevant_constraints = \
            get_relevant_constraints(cl, n_clusters, n_neighbors, nearest_centers)

        if log_performance_statistics:
            iteration_log['cl'] = len(relevant_constraints)

        # Add hard cannot-link constraints
        m.addConstrs(x[i, j] + x[i_, j] <= 1 for i, i_, j in zip(cl[idx_constraints, 0], cl[idx_constraints, 1],
                                                                 idx_clusters))

    # Add soft cannot-link constraints
    if len(scl) > 0:
        # Get relevant constraints
        idx_constraints, idx_clusters, relevant_constraints = \
            get_relevant_constraints(scl, n_clusters, n_neighbors, nearest_centers)

        if log_performance_statistics:
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

    # Add soft must-link constraints
    if len(sml) > 0:
        # Get relevant constraints
        idx_constraints, idx_clusters, relevant_constraints = \
            get_relevant_constraints(sml, n_clusters, n_neighbors, nearest_centers)

        if log_performance_statistics:
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

    # Each cluster must have at least one representative
    m.addConstrs(x.sum('*', j) >= 1 for j in range(n_clusters))

    # Set solver parameters
    if verbose < 2:
        m.setParam('OutputFlag', 0)
    if assignment_time_limit is not None:
        m._assignment_time_limit = assignment_time_limit
    if log_performance_statistics:
        iteration_log['cpu_model_setup'] = time.perf_counter() - tic_setup
    m.setParam('MIPFocus', 1)
    m.setParam('Presolve', 2)
    m.setParam('PrePasses', 1)
    if len(cl) == 0:
        m.setParam('TimeLimit', assignment_time_limit)
    else:
        m.setParam('TimeLimit', control_parameters['time_limit'])

    # Run optimization
    m.optimize(my_callback)

    # Check if feasible solution was found (continue optimization if total elapsed time does not exceed time limit)
    if m.SolCount == 0:
        return None, None, performance_log

    # Get labels from optimal assignment
    labels = np.array([j for i, j in distances if x[i, j].X > 0.5])

    # Get total penalty
    total_penalty = m.ObjVal - sum([v.Obj for v in x.values() if v.X > 0.5])

    if log_performance_statistics:
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
        performance_log['iterations'].append(iteration_log)

    return labels, total_penalty, performance_log


def get_relevant_constraints(constraints, n_clusters, n_neighbors, nearest_centers):
    """

    Args:
        constraints (numpy.ndarray): array that contains constraints on representatives as rows
        n_clusters (int): number of clusters
        n_neighbors (numpy.int64): number of nearest neighbors
        nearest_centers (numpy.ndarray): nearest centers (output from kd-tree)

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: idx_constraints, idx_clusters, relevant_constraints
    """
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


def get_total_distance(data, centers, labels, weights):
    """Computes total distance between objects and cluster centers

    Args:
        data (numpy.ndarray): feature vectors of representatives
        centers (numpy.ndarray): current positions of cluster centers
        labels (numpy.ndarray): current cluster assignments of objects
        weights (numpy.ndarray): number of objects that are represented by representatives

    Returns:
        numpy.float64: total distance
    """
    dist = (np.sqrt(((data - centers[labels, :]) ** 2).sum(axis=1)) * weights).sum()

    return dist


def drop_duplicate_hard_constraints(constraints):
    """

    Args:
        constraints (numpy.ndarray): array that contains constraints on representatives as rows

    Returns:
        numpy.ndarray: constraints without redundant constraints
    """
    if len(constraints) > 0:
        nodes_i = constraints.min(axis=1)
        nodes_j = constraints.max(axis=1)
        constraints = np.stack((nodes_i, nodes_j), axis=1)
        constraints = np.unique(constraints, axis=0)
    return constraints


def aggregate_soft_constraints(constraints, weights):
    """

    Args:
        constraints (numpy.ndarray): array that contains soft constraints on representatives as rows
        weights (numpy.ndarray): weights that correspond to soft constraints

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: constraints, weights
    """
    if len(constraints) > 0:
        nodes_i = constraints.min(axis=1)
        nodes_j = constraints.max(axis=1)
        constraints = np.stack((nodes_i, nodes_j), axis=1)
        constraints, ind = np.unique(constraints, axis=0, return_inverse=True)
        weights = pd.Series(weights).groupby(ind).sum().values
    return constraints, weights


def check_input(ml, cl, sml, scl, sml_weights, scl_weights, n_clusters, n_neighbors):
    """ Checks and cleans input

    Args:
        ml (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        cl (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        sml (List[Tuple[int, int]]): tuples of objects that are  subject to a soft must-link constraint
        scl (List[Tuple[int, int]]): tuples of objects that are  subject to a soft cannot-link constraint
        sml_weights (List[float]): weights associated with soft must-link constraints
        scl_weights (List[float]): weights associated with soft cannot-link constraints
        n_clusters (int): predefined number of clusters
        n_neighbors (int): objects will be assigned to one of the n_neighbors nearest cluster

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int]:
        cleaned input
    """
    # Initialize optional arguments
    if n_neighbors is None:
        n_neighbors = n_clusters
    else:
        n_neighbors = min(n_neighbors, n_clusters)
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
    """ Adjust value of control parameter n_neighbors

    Args:
        cl (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        n_clusters (int): predefined number of clusters
        n_neighbors (int): objects will be assigned to one of the n_neighbors nearest cluster

    Returns:
        Tuple[numpy.int64, numpy.int64]: new value of n_neighbors and max degree in respective graph
    """

    if len(cl) > 0:
        _, counts = np.unique(cl, return_counts=True)
        lower_bound = max(counts.max() + 1, n_neighbors)
        n_neighbors = min(lower_bound, n_clusters)
        return n_neighbors, counts.max()
    return n_neighbors, 0


def set_control_parameters(penalty, log_performance_statistics, time_limit, assignment_time_limit, tic, verbose):
    """ Puts all control parameters into a dictionary

    Args:
        penalty (str): strategy to compute penalty value
        log_performance_statistics (bool): if true, performance statistics are returned to user
        time_limit (float): overall time limit
        assignment_time_limit (None): solver time limit
        tic (float): start time
        verbose (int): verbosity

    Returns:
        Dict[str, str]: dictionary that contains all control parameters
    """

    # Set assignment time limit
    if (assignment_time_limit is None) and (time_limit is not None):
        assignment_time_limit = time_limit

    # Put all control parameters into a dictionary
    control_parameters = {'penalty': penalty, 'log_performance_statistics': log_performance_statistics,
                          'time_limit': time_limit, 'assignment_time_limit': assignment_time_limit,
                          'tic': tic, 'verbose': verbose}

    return control_parameters


def pccc(X, n_clusters, ml=None, cl=None, sml=None, scl=None, sml_weights=None,
         scl_weights=None, random_state=None, max_iter=1e10, n_neighbors=None, init='k-means++',
         penalty='avg_distance', return_all=False, log_performance_statistics=False, time_limit=None,
         assignment_time_limit=None, verbose=0):
    """Assigns cluster labels to objects subject to hard and soft must-link and cannot-link constraints

    Args:
        X (numpy.ndarray): feature vectors of objects
        n_clusters (int): predefined number of clusters
        ml (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        cl (List[Tuple[int, int]]): tuples of objects that are subject to a hard must-link constraint
        sml (List[Tuple[int, int]]): tuples of objects that are  subject to a soft must-link constraint
        scl (List[Tuple[int, int]]): tuples of objects that are  subject to a soft cannot-link constraint
        sml_weights (List[float]): weights associated with soft must-link constraints
        scl_weights (List[float]): weights associated with soft cannot-link constraints
        random_state (int): random state
        max_iter (float): maximum number of iterations of algorithm
        n_neighbors (int): objects will be assigned to one of the n_neighbors nearest cluster
        init (str): method to choose initial cluster positions (k-means++, random)
        penalty (str): strategy to compute penalty value (avg_distance, max_distance, quartile_distance)
        return_all (bool): if true, all stored information is returned to user
        log_performance_statistics (bool): if true, performance statistics are returned to user
        time_limit (None): overall time limit
        assignment_time_limit (None): solver time limit
        verbose (int): verbosity

    Returns:
        Tuple[]: depends on flags (return_all, log_performance_statistics)
    """

    # Set time limit
    if time_limit is None:
        time_limit = 1e7
    else:
        if assignment_time_limit is not None:
            assignment_time_limit = min(time_limit, assignment_time_limit)
        else:
            assignment_time_limit = time_limit

    # Start stopwatch
    tic = time.perf_counter()

    control_parameters = set_control_parameters(penalty, log_performance_statistics, time_limit,
                                                assignment_time_limit, tic, verbose)

    performance_log = {}
    if log_performance_statistics:
        performance_log['n_objects'] = X.shape[0]
        if ml is not None:
            performance_log['ml'] = len(ml)
        else:
            performance_log['ml'] = 0
        if cl is not None:
            performance_log['cl'] = len(cl)
        else:
            performance_log['cl'] = 0
        if sml is not None:
            performance_log['sml'] = len(sml)
        else:
            performance_log['sml'] = 0
        if scl is not None:
            performance_log['scl'] = len(scl)
        else:
            performance_log['scl'] = 0

    # Set initial parameters
    ml, cl, sml, scl, sml_weights, scl_weights, n_neighbors = check_input(ml, cl, sml, scl, sml_weights,
                                                                          scl_weights, n_clusters,
                                                                          n_neighbors)

    # Copy data
    data = X.astype(float).copy()

    if log_performance_statistics:
        tic_preprocessing = time.perf_counter()

    # Perform preprocessing
    data, weights, mapping, cl, sml, scl, confidence, kdt = preprocessing(data, ml, cl, sml, scl, sml_weights,
                                                                          scl_weights)

    # Perform feasibility check
    if data.shape[0] < n_clusters:
        warnings.warn('There is no feasible assignment!')
        if return_all:
            return None, None, data, weights, cl, None, performance_log
        else:
            if log_performance_statistics:
                return None, performance_log
            else:
                return None

    if log_performance_statistics:
        performance_log['cpu_preprocessing'] = time.perf_counter() - tic_preprocessing
        performance_log['n_ml_after_preprocessing'] = 0
        performance_log['n_cl_after_preprocessing'] = cl.shape[0]
        performance_log['n_sml_after_preprocessing'] = sml.shape[0]
        performance_log['n_scl_after_preprocessing'] = scl.shape[0]
        performance_log['n_representatives'] = data.shape[0]
        performance_log['n_neighbors'] = n_neighbors
        performance_log['iterations'] = []

    # Adjust n_neighbors based on hard cannot-link constraints
    n_neighbors, max_degree = adjust_n_neighbors(cl, n_clusters, n_neighbors)
    control_parameters['n_neighbors'] = n_neighbors

    # Choose initial cluster centers randomly
    centers = choose_initial_cluster_centers(data, n_clusters, random_state, init)

    initial_labels = None

    # Assign objects
    labels, total_penalty, performance_log = assign_objects(data, centers, weights, cl, sml, scl, confidence, kdt,
                                                            control_parameters, log_performance_statistics,
                                                            performance_log, labels=initial_labels)

    # Perform feasibility check
    if labels is None:
        warnings.warn('No feasible assignment found!')
        if return_all:
            return None, None, data, weights, cl, None, performance_log
        else:
            if log_performance_statistics:
                return None, performance_log
            else:
                return None

    # Initialize best labels
    best_labels = labels

    # Update centers
    centers = update_centers(data, centers, n_clusters, labels, weights)

    # Initialize best centers
    best_centers = centers.copy()

    # Compute solution quality
    best_solution_quality = get_total_distance(data, centers, labels, weights) + total_penalty

    n_iter = 0
    elapsed_time = time.perf_counter() - tic
    while (n_iter < max_iter) and (elapsed_time < time_limit):

        # Assign objects
        labels, total_penalty, performance_log = assign_objects(data, centers, weights, cl, sml, scl, confidence,
                                                                kdt, control_parameters, log_performance_statistics,
                                                                performance_log, labels=labels)

        # Perform feasibility check
        if labels is None:
            warnings.warn('There is no feasible assignment!')
            if return_all:
                return None, None, data, weights, cl, None, performance_log
            else:
                if log_performance_statistics:
                    return None, performance_log
                else:
                    return None

        # Update centers
        centers = update_centers(data, centers, n_clusters, labels, weights)

        # Compute solution quality
        solution_quality = get_total_distance(data, centers, labels, weights) + total_penalty

        # Check stopping criterion
        if solution_quality >= best_solution_quality:
            break
        else:
            # Update best labels and best total distance
            best_labels = labels
            best_centers = centers.copy()
            best_solution_quality = solution_quality

        # Increase iteration counter and compute elapsed time
        n_iter += 1
        elapsed_time = time.perf_counter() - tic

    if log_performance_statistics:
        performance_log['n_iter'] = n_iter + 2
        performance_log['n_neighbors_after_adjustment'] = n_neighbors
        performance_log['max_degree'] = max_degree
        performance_log['total_runtime'] = time.perf_counter() - tic

    if return_all:
        return best_labels[mapping], best_labels, data, weights, cl, best_centers, performance_log
    else:
        if log_performance_statistics:
            return best_labels[mapping], performance_log
        else:
            return best_labels[mapping]
