import numpy as np
from scipy.stats import skewnorm
from data_structures import Graph
#%%
def print_path(path):
    print('Path is: ' + ' '.join([str(node.node_id) for node in path]))
#%%
def generate_random_graph(node_count=10, density=0.5,distribution='normal', mean=10, std=3, skewness=0):
    edge_count = int(node_count*node_count*density)
    mask_indices = np.random.randint(low=0, high=node_count*node_count, size=edge_count)
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
