from collections import deque
import itertools
import copy
import sys
import time
import pickle

from scipy.optimize import linprog
import numpy as np
from tqdm import tqdm
from gurobipy import Model, GRB, LinExpr

from utils import construct_random_graph
#%%
# This function finds a path from source to target but it is not written in
# good shape. This why it is rewritten. Thus, it is obsolete.
def old_path_finder(graph, Delta):
    source_node, target_node = graph[1], graph[-1]
    # start from the source node's neighbors
    stack = set(source_node.get_adjacent_nodes(Delta))
    # instead of a queue we use set, which does not allow multiple occurences
    marked_nodes = set()
    
    # dictionary to store path
    incoming_nodes = {}
    for node in stack:
        incoming_nodes[node] = source_node    
    path_found = False
    # Continue iterations until no node is left or a path is found
    while len(stack) > 0 and not path_found:
        # pop a random node from the set. Thus, the algorithm is not BFS or DFS.
        next_node = stack.pop()
        # If it is marked/visited before, just skip
        if next_node in marked_nodes:
            continue
        # set the node as visited        
        marked_nodes.add(next_node)
        # get adjacent nodes and process them
        discovered_nodes = next_node.get_adjacent_nodes(Delta)
        for node in discovered_nodes:
            # no need to check if the discovered node is added to stack before.
            # set will handle thae case.
            stack.add(node)
            # if a path to a new node is found, update it.
            if node not in incoming_nodes:
                incoming_nodes[node] = next_node
            # if a path to target is found, just mark the flag and exit.
            if node.node_id == target_node.node_id:
                path_found = True
    
    # Keep backtracking for augmentation until source is reached.
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
        
    
        path.reverse()
        return path, delta
    return False, False
#%%
# This function finds a path from source to target and a modified version of the
# above one. It supports both DFS and BFS by a flag. Note that during the search,
# edges that have capacity below Delta is ignored to realize capacity scaling.
# So, neighbor of a node is defined as nodes that are connected with an arc larger
# Delta capacity
def find_path_from_source_to_target(graph, Delta, use_bfs=True):
    source_node, target_node = graph[1], graph[-1]
    # Put all neighbors of the source to start searching
    linked_list = deque(source_node.get_adjacent_nodes(Delta))
    # Use a boolean list to track which nodes were visited. Each index corresponds
    # to the node with the same id
    is_marked = [False] * (len(graph.node_list) + 1) 
    is_marked[1] = True # set source as visited
    # Track where any node is reached from
    where_path_came_from = {node: source_node for node in linked_list}
    
    path_found = False
    while linked_list and not path_found:
        if use_bfs:
            # pops the oldest element which makes it a queue
            current_node = linked_list.popleft()
        else:
            # pops the newest element which makes it a stack
            current_node = linked_list.pop()
        # Get neighbors wrt Delta
        discovered_nodes = current_node.get_adjacent_nodes(Delta)
        for node in discovered_nodes:
            node_id = node.node_id
            # If this node is marked before just ignore it.
            if not is_marked[node_id]:
                # appends to end of the linked_list
                linked_list.append(node)
                # If found a path to a new node, set its trace
                if node not in where_path_came_from:
                    where_path_came_from[node] = current_node
                # Found a path to target node set the flag and exit
                if node.node_id == target_node.node_id:
                    path_found = True
                is_marked[node_id] = True

    # Construct the path from the traces and set augment capacity delta during
    # the construction.
    if path_found:
        incoming_node = where_path_came_from[target_node]
        path = [target_node, incoming_node]
        delta = incoming_node.adjacency_map[target_node]
        while incoming_node != source_node:
            prev_incoming_node = incoming_node
            incoming_node = where_path_came_from[incoming_node]
            path.append(incoming_node)
            delta = min(delta,incoming_node.
                        adjacency_map[prev_incoming_node])
        
        path.reverse()
        return path, delta
    return False, False
#%%
# Augment delta units of flow along the given path and update the residual network.
# Update the heap of the graph if heap is used for capacity scaling.
def augment(graph, path, delta, update_heap):
    # For each node in the path, drop delta unit of capacity from the incoming arc
    # and add delta to the reverse direction
    for i in range(len(path)-1):
        from_node, to_node = path[i], path[i+1]
        from_node_id, to_node_id = from_node.node_id, to_node.node_id
        graph[from_node_id].adjacency_map[path[i+1]] -= delta
        graph[to_node_id].adjacency_map[path[i]] += delta
        
        # If heap is used, make the same modifications on the heap as well
        # Note that heap will auto-adjust itself.
        if update_heap:
            graph.edge_heap[(from_node_id, to_node_id)] -= delta
            if (to_node_id, from_node_id) in graph.edge_heap:
                graph.edge_heap[(to_node_id, from_node_id)] += delta
            else: 
                graph.edge_heap[(to_node_id, from_node_id)] = delta
            
#%%
# Compute the resulting flow in the system simply by summing flow on
# the reverse arcs of the target
def get_max_flow(graph):
    target_node = graph[-1]
    adjacency_map = target_node.adjacency_map
    return sum([flow for flow in adjacency_map.values()])

#%%
# This function is an implementation of capacity scaling that supports using
# heap for Delta update and bfs/dfs for path finding.
def solve_with_capacity_scaling(graph, use_heap=False, use_bfs=True):
    # Set initial Delta by its formula and continue until it is set to 0.
    U = graph.maximum_capacity
    Delta = 2**np.floor(np.log(U))
    Deltas,deltas = [],[]
    while Delta > 0:
        # Find a path and capacity of this path
        path, delta = find_path_from_source_to_target(graph, Delta, use_bfs)
        # Store Delta and deltas for further analysis
        Deltas.append(Delta)
        deltas.append(delta)
        # If a path is found, just augment and update delta otherwise
        if path:
            augment(graph, path, delta, update_heap=use_heap)
        else:
            if use_heap:
                # Get the heap that stores edges of the graph
                heap = copy.deepcopy(graph.edge_heap)
                # Pop all arcs that have 0 capacity
                while heap.peekitem()[1] == 0:
                    heap.popitem()
                # Pop elements until an edge is seen whose capacity is equal to
                # Delta than update Delta as the capacity of the edge that was
                # seen in the middle
                new_Delta = -1
                edges = []
                while new_Delta < Delta:
                    edge = heap.popitem()
                    new_Delta = edge[1]
                    edges.append(edge)
                new_Delta = int(edges[len(edges) // 2][1])
                if new_Delta < Delta:
                    Delta = new_Delta
                # Handle edge case where new_Delta == Delta == 1
                elif Delta == 1:
                    Delta = 0
                else:
                    # if Delta is 2 before, we want to make sure we do not skip
                    # any arc with capacity=1
                    Delta = 1
            else:
                Delta = int(Delta / 2)
    # Just return the max flow as an integer
    max_flow = get_max_flow(graph)
    return int(max_flow), Deltas, deltas
#%%
# This function converts the network flow problem to a linear programming problem
# that is compatible with scipy's linprog format.
def solve_with_scipy(graph):
    node_count = len(graph.node_list) - 1
    edge_list = graph.edge_list
    # Add an extra edge that connects target to source with infinite capacity.
    # It is added to be able to satisfy TUM constrint.
    edge_list.append((node_count,1,np.inf))
    edge_count = len(edge_list)
    # map each edge to a decision variable index
    edge_to_decision_var = {(t[0],t[1]): ind for ind,t in enumerate(edge_list)}
    # Note the extra node that is stored in the graph to label nodes starting
    # from 1.
    # Set coeffs of objective function to 0.
    c = np.zeros((edge_count))
    source_edges = [edge for edge in edge_list if edge[0] == 1]
    source_edge_indices = [edge_to_decision_var[(edge[0], edge[1])] for edge in source_edges]
    # Since linprog supports only minimization, set the OF s.t it minimizes the
    # negative outflow of the source.
    c[source_edge_indices] = -1
    
    # Set capacity constraints for each arc.
    b_ub = [e[2] for e in edge_list]
    A_ub = np.identity(edge_count)
    
    A_eq = np.zeros((node_count,edge_count))
    # Construct the flow balance matrix for each node
    for e in edge_list:
        var_id = edge_to_decision_var[e[0],e[1]]
        incoming_node_id = e[0]
        outgoing_node_id = e[1]
        A_eq[incoming_node_id - 1, var_id] = 1
        A_eq[outgoing_node_id - 1, var_id] = -1 

    b_eq = np.zeros((node_count,1))  
    # Solve the problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)     
    return int(-result.fun)
#%%
# This function converts the network flow problem to a linear programming problem
# that is compatible with gurobi format.
def solve_with_gurobi(graph):
    model = Model()
    model.setParam('OutputFlag',False)
    
    node_count = len(graph.node_list) - 1
    edge_list = graph.edge_list
    # Add the extra arc similar to above function
    edge_list.append((node_count,1,np.inf))
    edge_count = len(edge_list) 
    edge_to_decision_var = {(t[0],t[1]): ind for ind,t in enumerate(edge_list)}
    # Define the decision variables. Note that capacity constraints are 
    # addressed here as upper bounds, not separately
    model_vars = [model.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=capacity) for u,v,capacity in edge_list]
    
    objective_function = LinExpr()
    source_edges = [edge for edge in edge_list if edge[0] == 1]
    source_edge_indices = [edge_to_decision_var[(edge[0], edge[1])] for edge in source_edges]
    # Formulate the OF to maximize the outflow of the source.
    for ind in source_edge_indices:
        objective_function += model_vars[ind]
        
    model.setObjective(objective_function,GRB.MAXIMIZE)
    
    A_eq = np.zeros((node_count,edge_count))
    # Set up node balance constraints.
    for e in edge_list:
        var_id = edge_to_decision_var[e[0],e[1]]
        incoming_node_id = e[0]
        outgoing_node_id = e[1]
        A_eq[incoming_node_id - 1, var_id] = 1
        A_eq[outgoing_node_id - 1, var_id] = -1
    # Add the constraints to the model.
    for i in range(node_count):
        constraint = LinExpr()
        for j in range(len(edge_list)):
            constraint += A_eq[i,j] * model_vars[j]
        model.addConstr(constraint == 0)
    
    # Just optimize and return the flow
    model.optimize()
    return int(model.objVal)
#%%
def run_experiments():
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
    
    all_vars = {'heapt': heapt, 'regt': regt, 'scipyt': scipyt, 'gurobit': gurobit,
                'heap_Deltas': heap_Deltas, 'heap_deltas': heap_deltas, 
                'reg_Deltas': reg_Deltas, 'reg_deltas': reg_deltas}
    pickle.dump(all_vars, open('all_vars.pkl','wb'))
#%%
graph = construct_random_graph()
print(solve_with_capacity_scaling(copy.deepcopy(graph),use_bfs=False)[0])
print(solve_with_capacity_scaling(copy.deepcopy(graph), use_heap=True,use_bfs=False)[0])
print(solve_with_gurobi(copy.deepcopy(graph)))
print(solve_with_scipy(copy.deepcopy(graph)))