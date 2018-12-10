from collections import deque
import copy
import numpy as np
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


