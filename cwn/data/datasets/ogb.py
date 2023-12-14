import torch
import os.path as osp

from data.utils import convert_graph_dataset_with_rings
from data.datasets import InMemoryComplexDataset
from ogb.graphproppred import PygGraphPropPredDataset
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

class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, max_ring_size, use_edge_features=False, transform=None,
                 pre_transform=None, pre_filter=None, init_method='sum', simple=False, n_jobs=2):
        self.name = name
        
        self.use_lap = True # denote whether add the laplacian feature
        
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._simple = simple
        self._n_jobs = n_jobs
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=2, init_method=init_method, cellular=True)
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']
        
    @property
    def raw_file_names(self):
        name = self.name.replace('-', '_')  # Replacing is to follow OGB folder naming convention
        # The processed graph files are our raw files.
        return [f'{name}/processed/geometric_data_processed.pt']

    @property
    def processed_file_names(self):
        if self.use_lap:
            return [f'{self.name}_complex_new.pt', f'{self.name}_idx_new.pt', f'{self.name}_tasks_new.pt']
        else:
            return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']
    
    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and simple feats"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        suffix3 = "-S" if self._simple else ""
        return directory + suffix1 + suffix2 + suffix3

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)
    
    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        tasks = torch.load(self.processed_paths[2])
        return data, slices, idx, tasks
    
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
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)
        split_idx = dataset.get_idx_split()
        if self._simple:  # Only retain the top two node/edge features
            print('Using simple features')
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
        
        # NB: the init method would basically have no effect if 
        # we use edge features and do not initialize rings. 
        print(f"Converting the {self.name} dataset to a cell complex...")
        complexes, _, _ = convert_graph_dataset_with_rings(
            dataset,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_method=self._init_method,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs)
        
        if self.use_lap:
            complexes = self.cat_lap(complexes)
            
        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(split_idx, self.processed_paths[1])
        
        print(f'Saving num_tasks in {self.processed_paths[2]}...')
        torch.save(dataset.num_tasks, self.processed_paths[2])


def load_ogb_graph_dataset(root, name):
    raw_dir = osp.join(root, 'raw')
    dataset = PygGraphPropPredDataset(name, raw_dir)
    idx = dataset.get_idx_split()

    return dataset, idx['train'], idx['valid'], idx['test']
