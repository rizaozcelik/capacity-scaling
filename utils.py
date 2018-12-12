import json
import itertools

import numpy as np
from scipy.stats import skewnorm

from data_structures import Graph
from capacity_scaling import solve_with_capacity_scaling
from lp_solvers import solve_with_gurobi, solve_with_scipy
  
#%%
def print_path(path):
    print('Path is: ' + ' '.join([str(node.node_id) for node in path]))
#%%
def construct_random_graph(node_count=10, sparsity=0.5, distribution='normal', mean=10, std=3, skewness=0):
    mask_count = int(node_count*node_count*sparsity)
    mask_indices = np.random.randint(low=0, high=node_count*node_count, size=mask_count)
    mask_indices = np.unravel_index(mask_indices,(node_count,node_count))
    
    if distribution == 'normal':
        random_numbers = skewnorm.rvs(a=skewness, loc=mean, scale=std, size=(node_count,node_count))
    elif distribution == 'uniform':
        random_numbers = np.random.uniform(low=0,high=mean*2,size=(node_count, node_count))
    else:
        print('Not supported distribution', distribution)
        return None
    
    random_numbers[mask_indices] = 0
    # remove self loops
    np.fill_diagonal(random_numbers, 0)
    # mask source and target edges
    random_numbers[-1,:], random_numbers[:,0] = 0, 0
    random_numbers = np.floor(random_numbers)
    edge_list = [(i+1,j+1, random_numbers[i,j]) 
                    for i in range(node_count) for j in range(node_count) 
                    if random_numbers[i,j] > 0]
    return Graph(node_count, edge_list)
#%%
def construct_demo_graph():
    edge_list = [(1, 2, 5),
                 (1, 3, 8),
                 (1, 4, 3),
                 (1, 5, 9),
                 (2, 3, 2),
                 (2, 6, 12),
                 (3, 2, 5),
                 (3, 6, 2),
                 (4, 5, 3),
                 (4, 6, 7),
                 (5, 4, 6),
                 (5, 6, 4)]
    graph = Graph(6, edge_list)
    
    return graph
#%%
def parse_experiment_setup(experiment_setup_filepath):
    function_mappings = {'solve_with_capacity_scaling': solve_with_capacity_scaling,
                         'solve_with_gurobi': solve_with_gurobi,
                         'solve_with_scipy': solve_with_scipy}
    js = json.load(open(experiment_setup_filepath,'r'))
    statistical_params = js['statistical_params']
    node_counts = js['node_counts']
    sparsities = js['sparsities']
    solvers = js['solvers']
    solvers = [function_mappings[solver] for solver in solvers]
    solver_params = js['solver_params']
    solver_names = js['solver_names']
    experiment_name = js['write_filepath']
    graph_configs = list(itertools.product(node_counts, sparsities, 
                                           statistical_params))
    return graph_configs, solvers, solver_params, solver_names, experiment_name


    