import gc
import time

from tqdm import tqdm
from utils import construct_random_graph, parse_experiment_setup

from capacity_scaling import solve_with_capacity_scaling
from lp_solvers import solve_with_gurobi, solve_with_scipy
#%%
# This is a runner function that runs multiple experiments given solvers, their
# params and graph_configs. It saves the results to a csv.
def run_experiments(graph_configs, solvers, solver_params, solver_names,
                    filename=None, save_frequency=1):
    if not (len(solvers) == len(solver_params) == len(solver_names)):
        print('Error in params!')
        return
    
    with open(filename, 'w') as write_file:
        col_names = ['iter','name', 'node_count','density', 'distribution_name', 
                'mean', 'std', 'skewness', 'execution_time', 'Deltas', 'deltas']
        write_file.write('|'.join(col_names) + '\n')

    # A flag that will be used to stop experimenting when mismatched results are 
    # found
    mismatched_result = False
    # Each config has a format (node count, density, stats) where stats in another
    # tuple that contains distrubution name, mean, std, skewness.
    iter_count = -1
    for config in tqdm(graph_configs):
        node_count, density, statistical_params = config
        distribution_name, mean, std, skewness = statistical_params
        graph = construct_random_graph(node_count=node_count, density=density,   
                                      distribution=distribution_name,mean=mean,
                                      std=std,skewness=skewness)
        results = []
        if not mismatched_result:
            iter_count = iter_count + 1
            for solver,params, name in zip(solvers, solver_params, solver_names):
                gc.collect()
                try:
                    start = time.time()
                    result, Deltas, deltas = solver(graph.copy(),**params)
                    end = time.time()
                    execution_time = end - start
                    desired_statistics = [iter_count, name, node_count, density,
                                          distribution_name, mean, std, skewness,
                                          execution_time, Deltas, deltas]
                    
                    results.append(result)
                except Exception as e:
                    desired_statistics = [iter_count, name, node_count, density, 
                                          distribution_name, mean, std, skewness,
                                          'ERROR', 'ERROR', 'ERROR']
                if iter_count % save_frequency == 0:
                    desired_statistics = [str(stat) for stat in desired_statistics]
                    write_file = open(filename, 'a')
                    write_file.write('|'.join(desired_statistics) + '\n')
                    write_file.close()
            if len(set(results)) != 1:
                mismatched_result = True
                print('Incompatible result. Results are:', results)
        else:
            break

#%%
def main():
    config_path = './experiments/configs/experiment2_bfs_vs_dfs.json'
    run_experiments(*parse_experiment_setup(config_path))
#%%
if __name__ == '__main__':
    main()
