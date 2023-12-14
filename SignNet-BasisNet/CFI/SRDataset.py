import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import numpy as np
import networkx as nx
import pickle
import os
import scipy.io as sio
#from math import comb

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, split = 0):
        self.split = split
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr401224.g6", "sr361446.g6", "sr351899.g6", "sr351668.g6", "sr291467.g6", "sr281264.g6", "sr261034.g6", "sr251256.g6", "sr16622.g6"]  # sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data_' + str(self.split) + '.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        dataset = nx.read_graph6(self.raw_paths[self.split])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            #edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))
            edge_index = torch.tensor(list(datum.edges())).transpose(1, 0)
            data_list.append(Data(edge_index=edge_index, x=x, y=None))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])