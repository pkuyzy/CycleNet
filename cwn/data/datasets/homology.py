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

diameter = 10
node_num_large = 20
diameter_small = 1
node_num = 60
k_neighbors = 3
num_graphs = 500


# a big cycle and cycle_num small cycles
def generate_data():
    cycle_num = random.randint(2, 6)
    node_per_cycle = int(node_num / cycle_num)
    jitter = 0.05 * diameter / (node_num_large + node_num)
    X = [] # store the position of nodes
    center = [random.uniform(-diameter, diameter), random.uniform(-diameter, diameter)] # the center of the large cycle

    # generate the nodes for the large cycle
    angle_large = [2 * np.pi * i / node_num_large for i in range(node_num_large)]
    X += [[np.sin(a)*diameter/2 + center[0] + random.uniform(-jitter, jitter), np.cos(a)*diameter/2 + center[1] +  + random.uniform(-jitter, jitter)] for a in angle_large]

    # generate the center nodes for the small cycles
    #angles = [2 * np.pi * i / cycle_num for i in range(cycle_num)]
    angles = [2 * np.pi * i / cycle_num + random.uniform(-2 * np.pi / (6 * cycle_num), 2 * np.pi / (6 * cycle_num)) for i in range(cycle_num)]
    cycle_centers = [[np.sin(a)*diameter/2 + center[0] + random.uniform(-diameter_small / 4, diameter_small / 4), np.cos(a)*diameter/2 + center[1] +  + random.uniform(-diameter_small / 4, diameter_small / 4)] for a in angles]

    # generate the nodes for the small cycles
    for cc in cycle_centers:
        #angle_small = [2 * np.pi * i / node_per_cycle for i in range(node_per_cycle)]
        angle_small = [2 * np.pi * i / node_per_cycle + random.uniform(-2 * np.pi / (6 * node_per_cycle), 2 * np.pi / (6 * node_per_cycle)) for i in range(node_per_cycle)]
        X += [[np.sin(a)*diameter_small/2 + cc[0] + random.uniform(-jitter, jitter), np.cos(a)*diameter_small/2 + cc[1] +  + random.uniform(-jitter, jitter)] for a in angle_small]

    # generate the k-nearest neighbor graph for these nodes
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(X)

    G = nx.from_scipy_sparse_matrix(nbrs.kneighbors_graph(X, mode = 'distance'), parallel_edges=False)
    G.remove_edges_from(nx.selfloop_edges(G))


    data = Data()
    data.x = torch.zeros_like(torch.tensor(X)).float()
    data.edge_index = torch.LongTensor([e for e in G.edges()]).T
    data.edge_attr = None
    data.y = torch.tensor(cycle_num + 1).float().view(1, 1)

    return data
    
def generate_homology_data():
    np.random.seed(1234)
    random.seed(1234)
    graphs = [generate_data() for _ in range(num_graphs)]

    return graphs, list(range(int(0.8 * num_graphs))), list(range(int(0.8 * num_graphs), num_graphs)), list(range(int(0.8 * num_graphs), num_graphs))

class HomologyDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting Strongly Regular graphs."""

    def __init__(self, root, name, max_dim=2, num_classes=1, train_ids=None, val_ids=None, test_ids=None, 
                 include_down_adj=False, max_ring_size=None, n_jobs=2, init_method='sum'):
        self.name = name
        self._num_classes = 2
        self._n_jobs = n_jobs
        assert max_ring_size is None or max_ring_size > 3
        self._max_ring_size = max_ring_size
        cellular = (max_ring_size is not None)
        if cellular:
            assert max_dim == 2
        super(HomologyDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            include_down_adj=include_down_adj, cellular=cellular, init_method=init_method)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.train_ids = list(range(int(0.8 * self.len())))
        self.val_ids = list(range(int(0.8 * self.len()), self.len()))
        self.test_ids = list(range(int(0.8 * self.len()), self.len()))
        
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(HomologyDataset, self).processed_dir
        suffix = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix += f"_down_adj" if self.include_down_adj else ""
        return directory + suffix

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]       

    def process(self):
        
        graphs, _, _, _ = generate_homology_data()
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
