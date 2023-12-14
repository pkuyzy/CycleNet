import os
import torch
import pickle

from data.utils import compute_clique_complex_with_gudhi, compute_ring_2complex
from data.utils import convert_graph_dataset_with_rings, convert_graph_dataset_with_gudhi
from data.datasets import InMemoryComplexDataset
from definitions import ROOT_DIR
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import networkx as nx

import os.path as osp
import errno
import random
import numpy as np


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

num_graphs = 200
k = 4


def compute_v(b, k):
    v = np.zeros(k)
    for i in range(k):
        v[i] = int(b % 2)
        b = int(b / 2)
    return v

def cfi_graph(k = 3, l = 0):
    Nodes = []; A = []; V = []; cnt_node = 0
    for a in range(1, k + 2):
        for b in range(pow(2, k)):
            v = compute_v(b, k)
            if (a <= (k - l + 1) and v.sum() % 2 == 0) or (a >= (k - l + 2) and v.sum() % 2 == 1):
                V.append(v)
                A.append(a)
                Nodes.append(cnt_node)
                cnt_node += 1
    g = nx.Graph()
    g.add_nodes_from(Nodes)
    for i in range(len(Nodes) - 1):
        for j in range(i + 1, len(Nodes)):
            a1 = A[i]; v1 = V[i]; a2 = A[j]; v2 = V[j]
            assert a2 >= a1

            for m in range(1, k + 1):
                if (int(a2 % (k + 1)) == int((a1 + m)% (k + 1)) and v2[k - m] == v1[m - 1]):# or (int(a1 % k) == int((a2 + m)%k) and v1[k - m] == v2[m - 1]):
                    g.add_edge(i, j)
                    break
    return g

def generate_cfi_data():
    np.random.seed(1234)
    random.seed(1234)
    G1 = cfi_graph(k, 0); G2 = cfi_graph(k, 1)
    G1.remove_edges_from(nx.selfloop_edges(G1))
    G2.remove_edges_from(nx.selfloop_edges(G2))
    data_list = []
    node_index = [i for i in range(G1.number_of_nodes())]
    for i in range(int(num_graphs / 2)):
        random.shuffle(node_index)
        mapping = dict(zip(G1, node_index))
        G11 = nx.relabel_nodes(G1, mapping); G22 = nx.relabel_nodes(G2, mapping)
        #data1 = torch_geometric.utils.from_networkx(G11)
        data1 = Data()
        data1.x = torch.ones(G11.number_of_nodes(), 1).float()
        data1.edge_index = torch.LongTensor([e for e in G11.edges()]).T
        data1.y = torch.LongTensor([0])
        #data2 = torch_geometric.utils.from_networkx(G22)
        data2 = Data()
        data2.x = torch.ones(G22.number_of_nodes(), 1).float()
        data2.edge_index = torch.LongTensor([e for e in G22.edges()]).T
        data2.y = torch.LongTensor([1])
        data_list.append(data1)
        data_list.append(data2)

    #data.y = torch.tensor(cycle_centers).view(-1).float()
    #SCB = compute_shortest_cycle_basis(G)
    return data_list, list(range(int(0.8 * num_graphs))), list(range(int(0.8 * num_graphs), num_graphs)), list(range(int(0.8 * num_graphs), num_graphs))
   

class CFIDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting Strongly Regular graphs."""

    def __init__(self, root, name, max_dim=2, num_classes=2, train_ids=None, val_ids=None, test_ids=None, 
                 include_down_adj=False, max_ring_size=None, n_jobs=2, init_method='sum'):
        self.name = name
        self._num_classes = 2
        self._n_jobs = n_jobs
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2
        super(CFIDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            include_down_adj=include_down_adj, cellular=cellular, init_method=init_method)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.train_ids = list(range(int(0.8 * self.len())))
        self.val_ids = list(range(int(0.8 * self.len()), self.len()))
        self.test_ids = list(range(int(0.8 * self.len()), self.len()))
        
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(CFIDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        suffix += "_" + str(k)
        return directory + suffix

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]       

    def process(self):
        
        graphs, _, _, _ = generate_cfi_data()
        exp_dim = self.max_dim
        if self._cellular:
            print(f"Converting the {self.name} dataset to a cell complex...")
            complexes, max_dim, num_features = convert_graph_dataset_with_rings(
                graphs,
                max_ring_size=self._max_ring_size,
                include_down_adj=self.include_down_adj,
                init_method=self._init_method,
                init_edges=True,
                init_rings=True,
                n_jobs=self._n_jobs)
        else:
            print(f"Converting the {self.name} dataset with gudhi...")
            complexes, max_dim, num_features = convert_graph_dataset_with_gudhi(
                graphs,
                expansion_dim=exp_dim,                                               
                include_down_adj=self.include_down_adj,                    
                init_method=self._init_method)
        
        if self._max_ring_size is not None:
            assert max_dim <= 2
        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)
        
        # Now we save in opt format.
        path = self.processed_paths[0]
        torch.save(self.collate(complexes, self.max_dim), path)
