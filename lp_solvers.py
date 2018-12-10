from scipy.optimize import linprog
from gurobipy import Model, GRB, LinExpr
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
    return int(-result.fun), None, None
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
    return int(model.objVal), None, None