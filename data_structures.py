from collections import defaultdict
from heapdict import heapdict
class Edge:
    
    def __init__(self, incoming_node, outgoing_node, capacity):
        self.incoming_node = incoming_node
        self.outgoing_node = outgoing_node
        self.capacity = capacity
        
    def __lt__(self, other):
        self.capacity < other.capacity
        
    def __str__(self):
        return str(self.incoming_node) + ' ' + str(self.outgoing_node) + ' ' +               str(self.capacity)
    
class Node:
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.adjacency_map = defaultdict(int)
    
    def __str__(self):
        adjacency_str = ', '.join([str(node.node_id) + ': ' + str(capacity) for node, capacity in self.adjacency_map.items()])
        return 'Node id: ' + str(self.node_id) + '\n' + adjacency_str
    
    def add_edge(self, e):
        if e.incoming_node.node_id == self.node_id:
            self.adjacency_map[e.outgoing_node] = e.capacity
        else:
            print('Error in adding edge')
    
    def get_adjacent_nodes(self, delta=0):
        return [key for key,value in self.adjacency_map.items() if value >= delta]
            
class Graph:
    
    def __init__(self, node_count, edge_list):
        self.node_list = [Node(i) for i in range(node_count + 1)]
        self.maximum_capacity = -1
        self.edge_list = edge_list
        self.edge_heap = heapdict({(t[0], t[1]): t[2] for t in edge_list})
        for e in edge_list:
            self.maximum_capacity = max(self.maximum_capacity, e[2])
            incoming_node_id = e[0]
            self.node_list[incoming_node_id].add_edge(
                    Edge(self.node_list[e[0]], self.node_list[e[1]], e[2]))

    def __getitem__(self, index):
        return self.node_list[index]
    
    def copy(self):
        node_count = len(self.node_list) - 1
        edge_list_copy = self.edge_list.copy()
        return Graph(node_count, edge_list_copy)
        
        
        
        
        
        