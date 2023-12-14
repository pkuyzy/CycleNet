# generate data for shortest cycle basis

# The output should be a dict, every key correspond to a single graph. Every key contains a Cycle2Edge matrix.
import networkx as nx
import torch
import numpy as np
from torch_geometric.datasets import TUDataset, ZINC
import time
from ogb.graphproppred import PygGraphPropPredDataset
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops
import pickle

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

'''
def call(dataset, data_name, gn):
    dict_store = {}
    pbar_edge = tqdm(total=gn)
    for tt in range(gn):
        data = dataset[tt]
        g = nx.Graph()
        g.add_nodes_from([i for i in range(data.num_nodes)])
        ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
        ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                            range(np.shape(ricci_edge_index_)[1])]
        g.add_edges_from(ricci_edge_index)
        dict_store[tt] = \
            compute_shortest_cycle_basis(g)
        pbar_edge.update(1)

    pbar_edge.close()

    # original save name, for further evaluation
    save_name = './data/' + data_name + '_SL.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)


    return 0
if __name__ == "__main__":
    #d_names = ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY' ,'REDDIT-BINARY', 'ogbg-molhiv', 'ZINC']
    #GN = [188, 600, 1113, 5000, 1000, 2000, 41127, 10000]
    d_names = ['IMDB-BINARY' ,'REDDIT-BINARY', 'ogbg-molhiv', 'ZINC']
    GN = [1000, 2000, 41127, 10000]
    for (d_name, gn) in zip(d_names, GN):
        if d_name in ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY' ,'REDDIT-BINARY']:
            dataset = TUDataset('/data1/curvGN_LP/data/data/', d_name)
            save_name = dataset.name
            data_name = dataset.name
        elif d_name in ['ogbg-molhiv', 'ogbg-molpcba']:
            dataset = PygGraphPropPredDataset(name = d_name)
            save_name = d_name
            data_name = d_name
        elif d_name in ['ZINC']:
            dataset = ZINC(root = './dataset', subset = True)
            save_name = d_name
            data_name = d_name

        print(data_name)
        call(dataset, data_name, gn = gn)
'''