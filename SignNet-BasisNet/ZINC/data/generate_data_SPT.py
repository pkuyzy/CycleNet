# generate the shortest path tree cycle basis

# The output is a dict, every key corresponds to the cycle incidence matrix of each graph
import networkx as nx
import torch
import numpy as np
from torch_geometric.datasets import TUDataset, ZINC
import time
from ogb.graphproppred import PygGraphPropPredDataset
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops
import pickle
#from spectral import SpectralClustering
import random
import os

def get_edge_dict(g):
    edge_dict = {}
    edge_cnt = 0
    for edge in g.edges():
        if edge not in edge_dict:
            edge_dict[edge] = edge_cnt
            edge_dict[(edge[1], edge[0])] = edge_cnt
            edge_cnt += 1
    return edge_dict

def select_node_graph_random(g):
    center = []
    for sub_c in nx.connected_components(g):
        sub_g =  g.subgraph(sub_c)
        Nodes = [i for i in sub_c]
        Degree = [i[1] for i in sub_g.degree()]
        #center.append([random.choice(Nodes)])
        center.append([Nodes[np.argmax(Degree)]])
    center = np.array(center)
    return center.T

def select_node_graph_cluster(g, num_models = 10):
    # need to modify SpectralClustering
    center = []
    for sub_c in nx.connected_components(g):
        sub_g = g.subgraph(sub_c)
        adj_mat = nx.to_numpy_matrix(sub_g)
        n_cluster = num_models if len(sub_g.nodes()) >= num_models else len(sub_g.nodes())
        if len(sub_g.nodes()) < num_models:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        else:
            sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
        try:
            sc.fit(adj_mat)
        except BaseException:
            print("Do not find root node")
            return None

        tmp_center = np.array(np.array(sub_g.nodes())[sc.centers_].tolist() + [random.choice([__ for __ in sub_g.nodes()]) for _ in
                                                                  range(num_models - n_cluster)])
        center.append(tmp_center)
    center = np.array(center)
    return center.T

def compute_SPT_cycle(g, root_nodes, edge_dict, cnt_root):
    # compute the shortest path tree cycles, return the cycle incidence matrix (boundary matrix) and the length of all cycles

    # first compute the bfs tree, and get the father nodes for all nodes
    node2father = {}; mark_tree = {}; node2root = {}

    for ci, sub_c in enumerate(nx.connected_components(g)):
        sub_g = g.subgraph(sub_c)
        if root_nodes is not None:
            root = root_nodes[ci]
        else:
            sub_nodes = [sg for sg in sub_g.nodes()]

            # random choosing
            #root = sub_nodes[cnt_root] if cnt_root < len(sub_nodes) else random.choice(sub_nodes)
            # choosing with degree
            Degree = [i[1] for i in sub_g.degree()]
            root = sub_nodes[np.argsort(Degree)[-cnt_root]] if cnt_root < len(sub_nodes) else random.choice(sub_nodes)


        bfs_tree = list(nx.bfs_edges(sub_g, root))
        for tree_edge in bfs_tree:
            u, v = tree_edge
            node2father[v] = u
            mark_tree[(u, v)] = 1
            mark_tree[(v, u)] = 1
            node2root[u] = root
            node2root[v] = root
        node2father[root] = root
        node2root[root] = root

    # then get the paths of all cycles
    Cycle2edge = torch.zeros(len(g.edges()) - len(g.nodes()) + ci + 1, len(g.edges()))
    cnt_cycle = 0
    mark_nontree = [] # to mark which edge is non-tree, can help find the matching of cycles
    for edge in g.edges():
        if edge not in mark_tree:
            u, v = edge
            path = []; path_u = [u]; path_v = [v]
            # get path from u and v to root
            next_node = u
            while next_node != node2root[next_node]:
                next_node = node2father[next_node]
                path_u.append(next_node)
            next_node = v
            while next_node != node2root[next_node]:
                next_node = node2father[next_node]
                path_v.append(next_node)

            # compute the common ancestor, and get the loop
            len_u = len(path_u); len_v= len(path_v)
            if len_u > len_v:
                for v_i in range(len_v):
                    if path_u[v_i + len_u - len_v] == path_v[v_i]:
                        break
                u_i = v_i + len_u - len_v
            else:
                for u_i in range(len_u):
                    if path_u[u_i] == path_v[u_i + len_v - len_u]:
                        break
                v_i = u_i + len_v - len_u

            for i in range(0, u_i):
                tmp_edge = (path_u[i], path_u[i + 1])
                path.append(edge_dict[tmp_edge])
            for i in range(v_i , 0, -1):
                tmp_edge = (path_v[i], path_v[i - 1])
                path.append(edge_dict[tmp_edge])
            path.append(edge_dict[(u,v)])
            mark_nontree.append(edge_dict[u,v])
            Cycle2edge[cnt_cycle, path] = 1
            cnt_cycle += 1

    return Cycle2edge.T, mark_nontree

def compute_SPT_bases(g):
    edge_dict = get_edge_dict(g)
    g_clone = g.copy()
    SPT_cycle_bases, SPT_non_trees = compute_SPT_cycle(g_clone, None, edge_dict, 0)
    return SPT_cycle_bases, SPT_non_trees

def compute_SPT_bases_time(g):
    edge_dict = get_edge_dict(g)
    root_nodes = select_node_graph_random(g)
    for cnt_root in range(1):
        if root_nodes is not None:
            tmp_root = root_nodes[cnt_root].tolist()
        else:
            tmp_root = None
        g_clone = g.copy()
        compute_SPT_cycle(g_clone, tmp_root, edge_dict, cnt_root)
    return 0

def call(dataset, data_name, gn, root_num = 1):
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
        SPT_cycle_bases, SPT_non_trees = compute_SPT_bases(g, root_num)
        dict_store[tt] = [SPT_cycle_bases, SPT_non_trees, data.x, torch.LongTensor([edge for edge in g.edges()]).T]
        pbar_edge.update(1)

    pbar_edge.close()

    # original save name, for further evaluation
    save_name = './data/' + data_name + '_SPT_new.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(dict_store, f, pickle.HIGHEST_PROTOCOL)


    return 0

def evaluate_time(dataset, data_name, gn):
    pbar_edge = tqdm(total=gn)
    for tt in range(gn):
        data = dataset[tt]
        g = nx.Graph()
        g.add_nodes_from([i for i in range(data.num_nodes)])
        ricci_edge_index_ = np.array(remove_self_loops((data.edge_index.cpu()))[0])
        ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                            range(np.shape(ricci_edge_index_)[1])]
        g.add_edges_from(ricci_edge_index)
        compute_SPT_bases_time(g)
        pbar_edge.update(1)

    pbar_edge.close()
    return 0

if __name__ == "__main__":
    d_names = ['MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY' ,'REDDIT-BINARY', 'ogbg-molhiv', 'ZINC']
    GN = [188, 600, 1113, 5000, 1000, 2000, 41127, 10000]
    #d_names = ['ZINC']
    #GN = [10000]
    for (d_name, gn) in zip(d_names, GN):
        start = time.time()
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

        # for generation
        print(data_name)
        call(dataset, data_name, gn = gn)


        # for time evaluation
        #evaluate_time(dataset, data_name, gn = gn)
        #end = time.time()
        #log_info = "Dataset: {}, Generate Total Time: {}s, Average Time: {}s".format(data_name, end - start, (end - start) / gn)
        #print(log_info)
        #with open("./SPT_time.txt", "a") as f:
        #    f.write(log_info)
        #    f.write("\n")