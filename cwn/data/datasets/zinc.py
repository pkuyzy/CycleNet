import torch
import os.path as osp

from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from torch_geometric.datasets import ZINC
import networkx as nx
import numpy as np

def compute_lap(L):
    EigVal, EigVec = np.linalg.eigh(L)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    EigVal = EigVal.reshape(1, -1).repeat(len(EigVec), axis = 0)
    
    if len(EigVal) < 10:
        EigVec = np.pad(EigVec, ((0, 0), (0, 11 - len(EigVal))), 'constant', constant_values = (0, 0))
        EigVal = np.pad(EigVal, ((0, 0), (0, 11 - len(EigVal))), 'constant', constant_values = (0, 0))

    return np.concatenate((EigVal[:, :10], EigVec[:, :10]), axis=1)


class ZincDataset(InMemoryComplexDataset):
    """This is ZINC from the Benchmarking GNNs paper. This is a graph regression task."""

    def __init__(self, root, max_ring_size, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, subset=True, n_jobs=2):
        self.name = 'ZINC'
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._subset = subset
        self._n_jobs = n_jobs
        self.root = root
        
        self.use_lap = True # denote whether add the laplacian feature
        
        #self.download()
        super(ZincDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                          max_dim=2, cellular=True, num_classes=1)
        self.data, self.slices, idx = self.load_dataset()

        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

        self.num_node_type = 28
        self.num_edge_type = 4

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        name = self.name
        if self.use_lap:
            return [f'{name}_complex_new.pt', f'{name}_idx_new.pt']
        else:
            return [f'{name}_complex.pt', f'{name}_idx.pt']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        ZINC(self.raw_dir, subset=self._subset)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx
    
    
    def cat_lap(self, complexes):
        for i in range(len(complexes)):

            edge_index = complexes[i].cochains[0].upper_index
            G = nx.Graph()
            G.add_nodes_from(list(range(complexes[i].cochains[0].x.size(0))))
            G.add_edges_from(edge_index.T.tolist())
            B0 = nx.incidence_matrix(G, oriented=True)
  
            edge_index = complexes[i].cochains[1].upper_index
            if edge_index is not None:
                G = nx.Graph()
                G.add_nodes_from(list(range(complexes[i].cochains[1].x.size(0))))
                G.add_edges_from(edge_index.T.tolist())
                B1 = nx.incidence_matrix(G, oriented=True)
            else:
                B1 = None
            
            
            L0 = (B0 @ (B0.T)).todense()
            
            if B1 is None:
                L1 = ((B0.T) @ B0).todense()
            else:
                L1 = ((B0.T) @ B0 + B1 @ (B1.T)).todense()

            complexes[i].cochains[0].x = torch.cat((complexes[i].cochains[0].x, torch.from_numpy(compute_lap(L0))), dim = 1)
            complexes[i].cochains[1].x = torch.cat((complexes[i].cochains[1].x, torch.from_numpy(compute_lap(L1))), dim = 1)
        
        return complexes
        
    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing cell complex dataset for {self.name}")
        print(self.raw_dir)
        train_data = ZINC(self.raw_dir, subset=self._subset, split='train')
        val_data = ZINC(self.raw_dir, subset=self._subset, split='val')
        test_data = ZINC(self.raw_dir, subset=self._subset, split='test')

        data_list = []
        idx = []
        start = 0
        print("Converting the train dataset to a cell complex...")
        train_complexes, _, _ = convert_graph_dataset_with_rings(
            train_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        
        if self.use_lap:
            train_complexes = self.cat_lap(train_complexes)

        data_list += train_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the validation dataset to a cell complex...")
        val_complexes, _, _ = convert_graph_dataset_with_rings(
            val_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
            
        if self.use_lap:
            val_complexes = self.cat_lap(val_complexes)
        data_list += val_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the test dataset to a cell complex...")
        test_complexes, _, _ = convert_graph_dataset_with_rings(
            test_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        if self.use_lap:
            test_complexes = self.cat_lap(test_complexes)
        data_list += test_complexes
        idx.append(list(range(start, len(data_list))))

        path = self.processed_paths[0]
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(data_list, 2), path)
        
        path = self.processed_paths[1]
        print(f'Saving idx in {path}....')
        torch.save(idx, path)

    @property
    def processed_dir(self):
        """Overwrite to change name based on edges"""
        directory = super(ZincDataset, self).processed_dir
        suffix0 = "_full" if self._subset is False else ""
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix0 + suffix1 + suffix2


def load_zinc_graph_dataset(root, subset=True):
    raw_dir = osp.join(root, 'ZINC', 'raw')

    train_data = ZINC(raw_dir, subset=subset, split='train')
    val_data = ZINC(raw_dir, subset=subset, split='val')
    test_data = ZINC(raw_dir, subset=subset, split='test')
    data = train_data + val_data + test_data

    if subset:
        assert len(train_data) == 10000
        assert len(val_data) == 1000
        assert len(test_data) == 1000
    else:
        assert len(train_data) == 220011
        assert len(val_data) == 24445
        assert len(test_data) == 5000

    idx = []
    start = 0
    idx.append(list(range(start, len(train_data))))
    start = len(train_data)
    idx.append(list(range(start, start + len(val_data))))
    start = len(train_data) + len(val_data)
    idx.append(list(range(start, start + len(test_data))))

    return data, idx[0], idx[1], idx[2]


