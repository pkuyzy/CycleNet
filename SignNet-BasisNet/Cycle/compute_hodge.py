# generate data for shortest cycle basis

# The output should be a dict, every key correspond to a single graph. Every key contains a Cycle2Edge matrix.
import networkx as nx
import torch
import numpy as np
import os
import networkx as nx

def hodge_positional_encoding(data):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    g = nx.Graph()
    g.add_nodes_from([i for i in range(data.num_nodes)])
    g.add_edges_from([(i[0].item(), i[1].item())for i in data.edge_index.T])

    B = nx.incidence_matrix(g, oriented=True)
    L1 = ((B.T) @ B).todense()

    EigVal, EigVec = np.linalg.eigh(L1)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    thres = EigVal <= 1e-5
    #L1_cycle = torch.from_numpy(EigVec[:, thres]).float()
    #L1_noncycle = torch.from_numpy(EigVec[:, ~thres]).float()
    L1_cycle = torch.from_numpy(EigVec[:, thres] @ EigVec[:, thres].T).float()
    L1_noncycle = torch.from_numpy(EigVec[:, ~thres] @ EigVec[:, ~thres].T).float()

    # add the information of 2-cells
    SCB = compute_shortest_cycle_basis(g.to_undirected())
    #beta, le = SCB.shape
    #SCB = np.concatenate((SCB.reshape(beta, le, 1), SCB.reshape(beta, le, 1)), axis = 2).reshape(beta, 2 * le)

    # load the information of 2-cells (shortest cycle basis)
    cell_L1 = L1 + SCB.T @ SCB

    maxeigcell = np.linalg.eigvalsh(cell_L1)[-1]
    if maxeigcell > 1e-5:
        cell_L1 = cell_L1 / maxeigcell


    maxeig = np.linalg.eigvalsh(L1)[-1]
    if maxeig > 1e-5:
        L1 = L1 / maxeig

    assert len(L1_cycle) == data.edge_index.size(1)
    return torch.tensor(L1_cycle), torch.from_numpy(L1).to_sparse(), torch.from_numpy(SCB).to_sparse(), torch.from_numpy(cell_L1).to_sparse()

def compute_hodge_encodings(train_data, val_data, test_data, data_name):
    if not os.path.exists("./hodge"):
        os.mkdir("./hodge")
    if os.path.exists("./hodge/" + data_name + "_train.pt"):
        train_hodge = torch.load("./hodge/" + data_name + "_train.pt")
    else:
        train_hodge = [hodge_positional_encoding(g) for g in train_data]
        torch.save(train_hodge, "./hodge/" + data_name + "_train.pt")
    if os.path.exists("./hodge/" + data_name + "_val.pt"):
        val_hodge = torch.load("./hodge/" + data_name + "_val.pt")
    else:
        val_hodge = [hodge_positional_encoding(g) for g in val_data]
        torch.save(val_hodge, "./hodge/" + data_name + "_val.pt")
    if os.path.exists("./hodge/" + data_name + "_test.pt"):
        test_hodge = torch.load("./hodge/" + data_name + "_test.pt")
    else:
        test_hodge = [hodge_positional_encoding(g) for g in test_data]
        torch.save(test_hodge, "./hodge/" + data_name + "_test.pt")

    return train_hodge, val_hodge, test_hodge

def get_edge_dict(g):
    edge_dict = {}
    edge_cnt = 0
    for edge in g.edges():
        if edge not in edge_dict:
            edge_dict[edge] = edge_cnt
            edge_dict[(edge[1], edge[0])] = edge_cnt
            edge_cnt += 1
    return edge_dict

def compute_SPT_cycle(g, root, edge_dict):
    # compute the shortest path tree cycles, return the cycle incidence matrix (boundary matrix) and the length of all cycles
    cycle_path = []; cycle_length = []; cycle_path_direct = []

    # first compute the bfs tree, and get the father nodes for all nodes
    node2father = {}; mark_tree = {}
    bfs_tree = list(nx.bfs_edges(g, root))
    for tree_edge in bfs_tree:
        u, v = tree_edge
        node2father[v] = u
        mark_tree[(u, v)] = 1
        mark_tree[(v, u)] = 1
    node2father[root] = root

    # then get the paths of all cycles, here do not add cycles that path_u and path_v has the same edges
    for edge in g.edges():
        if edge not in mark_tree:
            u, v = edge
            path = []; path_u = [u]; path_v = [v]; path_direct = []
            # get path from u and v to root
            next_node = u
            while next_node != root:
                next_node = node2father[next_node]
                path_u.append(next_node)
            next_node = v
            while next_node != root:
                next_node = node2father[next_node]
                path_v.append(next_node)

            # Here u and v cannot be the root node, else, there will be no generated cycles.
            if path_u[-2] == path_v[-2]:
                continue
            else:
                path_v.reverse()
                path += [edge_dict[(path_u[cp], path_u[cp + 1])] for cp in range(len(path_u) - 1)]
                path += [edge_dict[(path_v[cp], path_v[cp + 1])] for cp in range(len(path_v) - 1)]
                path += [edge_dict[(u, v)]]
                path_direct += [(edge_dict[(path_u[cp], path_u[cp + 1])] + 1e-5) * pow(-1, cp + 1) for cp in range(len(path_u) - 1)]
                path_direct += [(edge_dict[(path_v[cp], path_v[cp + 1])] + 1e-5) * pow(-1, cp + 1) for cp in range(len(path_v) - 1)]
                path_direct += [edge_dict[(u, v)] + 1e-5]
                cycle_path.append(path)
                cycle_length.append(len(path))
                cycle_path_direct.append(path_direct)
    return cycle_path, cycle_length, cycle_path_direct


def compute_shortest_cycle_basis(g):
    edge_dict = get_edge_dict(g)

    # set all nodes as the root, and get the corresponding cycle basis
    total_cycle_path = []; total_cycle_length = []; total_cycle_path_direct = []
    len_c = 0
    for sub_c in nx.connected_components(g):
        len_c += 1
        sub_g = g.subgraph(sub_c)
        for root in sub_g.nodes():
            cycle_path, cycle_length, cycle_path_direct = compute_SPT_cycle(sub_g, root, edge_dict)
            total_cycle_path += cycle_path; total_cycle_length += cycle_length; total_cycle_path_direct += cycle_path_direct

    # sort the cycles according to the length of cycles, and use the greedy algorithm to find the shortest cycle basis
    sort_index = np.argsort(total_cycle_length)
    mark_low = torch.zeros(len(g.edges)) - 1 # stores the index in the shortest_cycle_path
    shortest_cycle_path = []; shortest_cycle_path_reduction = [] # each stores the original shortest cycle basis, and the basis after reduction
    for i in sort_index:
        tmp_cycle = total_cycle_path[i]
        while mark_low[max(tmp_cycle)] != -1:
            mi = int(mark_low[max(tmp_cycle)].item())
            cycle_union = set(tmp_cycle + shortest_cycle_path_reduction[mi])
            cycle_inter = set(tmp_cycle) & set(shortest_cycle_path_reduction[mi])
            tmp_cycle = list(cycle_union.difference(cycle_inter))
            if len(tmp_cycle) == 0:
                break
        if len(tmp_cycle) > 0:
            shortest_cycle_path.append(total_cycle_path_direct[i])
            shortest_cycle_path_reduction.append(tmp_cycle)
            mark_low[max(tmp_cycle)] = len(shortest_cycle_path) - 1

    # initialize the final cycle incidence matrix
    beta = len(g.edges()) - len(g.nodes()) + len_c
    shortest_cycle_basis = np.zeros((beta, len(g.edges())))
    assert beta == len(shortest_cycle_path)
    for cs, scp in enumerate(shortest_cycle_path):
        scp = np.array(scp)
        scp_int = scp.astype(int)
        #print(scp)
        #print(scp_int)
        shortest_cycle_basis[cs, scp_int[scp>=0]] = 1
        shortest_cycle_basis[cs, -scp_int[scp<0]] = -1
        #print(shortest_cycle_basis[cs])
    return shortest_cycle_basis#.to_sparse()
