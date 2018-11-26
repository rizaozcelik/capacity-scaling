import sys
import random
import copy
from data_structures import Graph
from scipy.optimize import linprog
import numpy as np
from tqdm import tqdm
from utils import construct_demo_graph, generate_random_graph, print_path
from baseline import solve_ff
import time
import itertools
from gurobipy import *
#%%
def find_path_from_source_to_target(graph, Delta):
    source_node, target_node = graph[1], graph[-1]
    stack = set(source_node.get_adjacent_nodes(Delta))
    iter_count = 0
    marked_nodes = set()
    
    incoming_nodes = {}
    for node in stack:
        incoming_nodes[node] = source_node    
#        print(node.node_id)
#    print('initial stack', [node.node_id for node in stack])
    path_found = False
    while len(stack) > 0 and not path_found:
        iter_count = iter_count + 1
    #    print(list(stack), list(marked_nodes))
        next_node = stack.pop()
        if next_node in marked_nodes:
            continue
#        print(next_node)
        
        discovered_nodes = [] # empty the list in case condition is not true
#        if next_node in graph: # dead end check
        discovered_nodes = next_node.get_adjacent_nodes(Delta)
#        print('added', next_node.node_id)
#        print('discovered', [node.node_id for node in discovered_nodes])
        marked_nodes.add(next_node)
    #    print(marked_nodes)
        for node in discovered_nodes:
            stack.add(node)
            if node not in incoming_nodes:
                incoming_nodes[node] = next_node
            if node.node_id == target_node.node_id:
#                print('dasdasdad')
                path_found = True
        if iter_count > 10000:
            print('Entered an infinite loop in path finding')
            sys.exit(0)            
            break
    
    iter_count = 0
#    print(path_found)
    if path_found:
        incoming_node = incoming_nodes[target_node]
        path = [target_node, incoming_node]
        delta = incoming_node.adjacency_map[target_node]
        while incoming_node != source_node:
            prev_incoming_node = incoming_node
            incoming_node = incoming_nodes[incoming_node]
            path.append(incoming_node)
            delta = min(delta,incoming_node.
                        adjacency_map[prev_incoming_node])
        
            if iter_count > 10000:
                print('Infinite loop in backtracking')
                sys.exit(0)
                break
    
        path.reverse()
        return path, delta
    return False, False
#%%
def augment(graph, path, delta, update_heap):
    for i in range(len(path)-1):
        from_node, to_node = path[i], path[i+1]
        from_node_id, to_node_id = from_node.node_id, to_node.node_id
    #    print(from_node_id, to_node_id)
        graph[from_node_id].adjacency_map[path[i+1]] -= delta
        graph[to_node_id].adjacency_map[path[i]] += delta
        
        if update_heap:
            graph.edge_heap[(from_node_id, to_node_id)] -= delta
            if (to_node_id, from_node_id) in graph.edge_heap:
                graph.edge_heap[(to_node_id, from_node_id)] += delta
            else: 
                graph.edge_heap[(to_node_id, from_node_id)] = delta
            
#        print(graph[from_node_id])
#%%
def get_max_flow(graph):
    target_node = graph[-1]
    adjacency_map = target_node.adjacency_map
    return sum([flow for flow in adjacency_map.values()])

#%%
def solve_with_capacity_scaling(graph, use_heap=False):
    U = graph.maximum_capacity
    Delta = 2**np.floor(np.log(U))
    Deltas,deltas = [],[]
    while Delta > 0:
        path, delta = find_path_from_source_to_target(graph, Delta)
        Deltas.append(Delta)
        deltas.append(delta)
        if path:
            augment(graph, path, delta, update_heap=use_heap)
        else:
            if use_heap:
                heap = copy.deepcopy(graph.edge_heap)
                while heap.peekitem()[1] == 0:
                    heap.popitem()
                new_Delta = -1
                edges = []
                while new_Delta < Delta:
                    edge = heap.popitem()
                    new_Delta = edge[1]
                    edges.append(edge)
#                print(edges)
                new_Delta = int(edges[len(edges) // 2][1])
                if new_Delta < Delta:
                    Delta = new_Delta
                elif Delta == 1:
                    Delta = 0
                else:
                    Delta = 1
#                Delta = int(Delta / 2)
#                print(new_Delta, Delta)
            else:
                Delta = int(Delta / 2)
#        break
    max_flow = get_max_flow(graph)
#    print(max_flow, graph[-1])
#    print(max_flow)
    return int(max_flow), Deltas, deltas
#%%
def solve_with_scipy(graph):
    edge_list = graph.edge_list
    edge_to_decision_var = {(t[0],t[1]): ind for ind,t in enumerate(edge_list)}
    node_count = len(graph.node_list) - 1
    c = np.zeros((len(edge_list) + 1))
    # maximize the outflow of s
    source_edges = [edge for edge in edge_list if edge[0] == 1]
    source_edge_indices = [edge_to_decision_var[(edge[0], edge[1])] for edge in source_edges]
    c[source_edge_indices] = -1
    
    b_ub = [e[2] for e in edge_list] + [np.inf]
    A_ub = np.identity(len(edge_list) + 1)
    
    A_eq = np.zeros((node_count,len(edge_list) + 1))
    # for extra edge that connects target to source
    A_eq[0, -1], A_eq[-1,-1] = -1, 1 
    
    for e  in edge_list:
        var_id = edge_to_decision_var[e[0],e[1]]
        incoming_node_id = e[0]
        outgoing_node_id = e[1]
        A_eq[incoming_node_id - 1, var_id] = 1
        A_eq[outgoing_node_id - 1, var_id] = -1 

    b_eq = np.zeros((node_count,1))  
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)     
    return int(-res.fun)
#%%
def solve_with_gurobi(graph):
    model = Model()
    model.setParam('OutputFlag',False)
    
    node_count = len(graph.node_list) - 1
    edge_list = graph.edge_list
    edge_list.append((node_count,1,np.inf))
    edge_count = len(edge_list) 
    edge_to_decision_var = {(t[0],t[1]): ind for ind,t in enumerate(edge_list)}
    
    model_vars = [model.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=capacity) for u,v,capacity in edge_list]
    
    objective_function = LinExpr()
    source_edges = [edge for edge in edge_list if edge[0] == 1]
    source_edge_indices = [edge_to_decision_var[(edge[0], edge[1])] for edge in source_edges]
    for ind in source_edge_indices:
        objective_function += model_vars[ind]
        
    model.setObjective(objective_function,GRB.MAXIMIZE)
    
    A_eq = np.zeros((node_count,edge_count))
    # for extra edge that connects target to source
    A_eq[0, -1], A_eq[-1,-1] = -1, 1 
    for e in edge_list:
        var_id = edge_to_decision_var[e[0],e[1]]
        incoming_node_id = e[0]
        outgoing_node_id = e[1]
        A_eq[incoming_node_id - 1, var_id] = 1
        A_eq[outgoing_node_id - 1, var_id] = -1
    
    for i in range(node_count):
        constraint = LinExpr()
        for j in range(len(edge_list)):
            constraint += A_eq[i,j] * model_vars[j]
        model.addConstr(constraint == 0)
    
    model.optimize()
    return int(model.objVal)
#%%
random.seed(0)
configs = [('normal',1000,50,-10),
                      ('normal',1000,50,10),
                      ('normal',1000,50,0),
                      ('normal',1000,500,-10),
                      ('normal',1000,500,10),
                      ('normal',1000,500,0),
                      ('normal',50,3,-10),
                      ('normal',50,3,10),
                      ('normal',50,3,0),
                      ('normal',50,30,-10),
                      ('normal',50,30,10),
                      ('normal',50,30,0),
                      ('uniform',1000,None,None),
                      ('uniform',50,None,None)]
node_counts = [50, 150, 300]
densities = [0.2, 0.5, 0.8]
#node_counts = [50]
#densities = [0.2]
experimental_setup = list(itertools.product(node_counts, densities, configs))
#%%
heapt, regt, scipyt, gurobit = [], [], [], []   
heap_Deltas, heap_deltas, reg_Deltas, reg_deltas = [], [], [], []
for setup in tqdm(experimental_setup):
    node_count, density, configs = setup
    distribution, mean, std, skewness = configs
    graph = generate_random_graph(node_count=node_count, density=density,   
                                  distribution=distribution,mean=mean,
                                  std=std,skewness=skewness)

    t0 = time.time()
    heap_res, heap_Delta, heap_delta = \
    solve_with_capacity_scaling(copy.deepcopy(graph), use_heap=True)
    t1 = time.time()
    heapt.append(t1 - t0)
    heap_Deltas.append(heap_Delta)
    heap_deltas.append(heap_delta)    
    
    reg_res, reg_Delta, reg_delta = \
    solve_with_capacity_scaling(copy.deepcopy(graph))
    t2 = time.time()
    regt.append(t2 - t1)
    reg_Deltas.append(reg_Delta)
    reg_deltas.append(reg_delta)    
    
    lin_res = solve_with_scipy(copy.deepcopy(graph))
    t3 = time.time()
    scipyt.append(t3 - t2)
    
    gurobi_res = solve_with_gurobi(copy.deepcopy(graph))
    t4 = time.time()
    gurobit.append(t4 - t3)
    
    if not(gurobi_res == reg_res == heap_res == lin_res):
        print('ups')
        break
    
    