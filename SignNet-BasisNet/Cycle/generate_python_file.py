import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
import random
import torch_geometric
from sign_net.model import GNN
from sign_net.cyclenet import CycleNet, CycleNet_edge, CycleNet_Hodge_original as CycleNet_Hodge, CycleNet_Hodge_edge
from sign_net.sign_net import SignNetGNN
from sign_net.transform import EVDTransform
from compute_hodge import compute_shortest_cycle_basis, hodge_positional_encoding
from torch_geometric.data import InMemoryDataset
import time
from shutil import copy, rmtree

#data_list = [Data(...), ..., Data(...)]
#loader = DataLoader(data_list, batch_size=32)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



set_seed(1234) # to generate data
diameter = 10
node_num_large = 20#100
diameter_small = 1
node_num = 60#300
k_neighbors = 3
num_graphs = 500
batch_size = 16
epochs = 200
hidden_dim = 128
device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
model_name = 'SignNet' # can be GNN/SignNet/CycleNet/CycleNet_edge/CycleNet_Hodge/CycleNet_Hodge_edge
use_hodge = True if model_name == 'CycleNet' or model_name == 'CycleNet_edge' or model_name == 'CycleNet_Hodge' or model_name == 'CycleNet_Hodge_edge' else False

save_appendix = time.strftime("%Y%m%d%H%M%S")
res_dir = 'results/' + save_appendix
print('Results will be saved in ' + res_dir)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# Backup python files.
copy('generate_python_file.py', res_dir)
copy('sign_net/model.py', res_dir)
copy('sign_net/cyclenet.py', res_dir)



def generate_data():
    cycle_num = random.randint(2, 5)
    node_per_cycle = int(node_num / cycle_num)
    jitter = 0.05 * diameter / (node_num_large + node_num)
    X = [] # store the position of nodes
    center = [random.uniform(-diameter, diameter), random.uniform(-diameter, diameter)] # the center of the large cycle

    # generate the nodes for the large cycle
    angle_large = [2 * np.pi * i / node_num_large for i in range(node_num_large)]
    X += [[np.sin(a)*diameter/2 + center[0] + random.uniform(-jitter, jitter), np.cos(a)*diameter/2 + center[1] +  + random.uniform(-jitter, jitter)] for a in angle_large]

    # generate the center nodes for the small cycles
    angles = [2 * np.pi * i / cycle_num for i in range(cycle_num)]
    cycle_centers = [[np.sin(a)*diameter/2 + center[0] + random.uniform(-diameter_small / 4, diameter_small / 4), np.cos(a)*diameter/2 + center[1] +  + random.uniform(-diameter_small / 4, diameter_small / 4)] for a in angles]

    # generate the nodes for the small cycles
    for cc in cycle_centers:
        angle_small = [2 * np.pi * i / node_per_cycle for i in
                       range(node_per_cycle)]
        X += [[np.sin(a)*diameter_small/2 + cc[0] + random.uniform(-jitter, jitter), np.cos(a)*diameter_small/2 + cc[1] +  + random.uniform(-jitter, jitter)] for a in angle_small]

    # generate the k-nearest neighbor graph for these nodes
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(X)

    G = nx.from_scipy_sparse_matrix(nbrs.kneighbors_graph(X, mode = 'distance'), parallel_edges=False)
    G.remove_edges_from(nx.selfloop_edges(G))


    #data = torch_geometric.utils.from_networkx(G)
    data = Data()
    data.x = torch.zeros(len(X), 2).float()
    data.edge_index = torch.LongTensor([e for e in G.edges()]).T
    data.edge_attr = torch.zeros(len(G.edges()), 1).float()
    data.y = torch.tensor(cycle_num + 1).float()
    return data


class Homology_Data(InMemoryDataset):
    def __init__(self, root = "./data", transform=None, pre_transform=None, pre_filter=None, split = 'train'):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return 0#['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_test.pt']

    #def download(self):
        # Download to `self.raw_dir`.
    #    return 0

    def process(self):
        if self.split == "train":
            data_list = [generate_data() for _ in range(int(0.8 * num_graphs))]
        elif self.split == "test":
            data_list = [generate_data() for _ in range(int(0.2 * num_graphs))]
        else:
            return NotImplementedError

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        if self.split == 'train':
            torch.save((data, slices), self.processed_paths[0])
        elif self.split == "test":
            torch.save((data, slices), self.processed_paths[1])
        else:
            return NotImplementedError



transform = EVDTransform('sym')
train_datasets = Homology_Data(split = 'train', transform=transform)
test_datasets = Homology_Data(split = 'test', transform=transform)

train_index = [i for i in range(int(0.8 * num_graphs))]
#train_loader = DataLoader(train_datasets, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_datasets, shuffle = False, batch_size = batch_size)

if use_hodge:
    train_file = "./data/train_hodge.pt"; test_file = "./data/test_hodge.pt"
    if not os.path.exists(train_file):
        print("computing train hodge")
        train_hodge = [hodge_positional_encoding(data) for data in train_datasets]
        torch.save(train_hodge, train_file)
    else:
        train_hodge = torch.load(train_file)
    if not os.path.exists(test_file):
        print('computing test hodge')
        test_hodge = [hodge_positional_encoding(data) for data in test_datasets]
        torch.save(test_hodge, test_file)
    else:
        test_hodge = torch.load(test_file)
else:
    train_hodge = None; test_hodge = None


curves = []
average_time = 0
for seed in range(5):
    t1 = time.time()
    set_seed(seed)
    # model = GNN(2, 1, 128, 2 * cycle_num, nlayer=5, gnn_type='GINEConv', pooling = 'mean')
    if model_name == 'GNN':
        model = GNN(2, 1, hidden_dim, 1, nlayer=5, gnn_type='GINEConv')
    elif model_name == 'CycleNet':
        model = CycleNet(2, 1 + hidden_dim, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv')
        # model = CycleNet(2, 1, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
    elif model_name == 'CycleNet_Hodge':
        model = CycleNet_Hodge(2, 1 + hidden_dim, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv')
    elif model_name == 'CycleNet_edge':
        model = CycleNet_edge(2, hidden_dim, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv')
    elif model_name == 'CycleNet_Hodge_edge':
        model = CycleNet_edge(2, hidden_dim, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv')
    elif model_name == 'SignNet':
        model = SignNetGNN(2, 1, n_hid=hidden_dim, n_out=1, nl_signnet=5, nl_gnn=5, nl_rho=8, ignore_eigval=False,
                           gnn_type='GINEConv')
    else:
        raise NotImplementedError
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10,
                                                           min_lr=1e-6)
    criterion = torch.nn.MSELoss()
    test_criterion = torch.nn.L1Loss()
    best_test_score = 1e10

    for i in range(epochs):
        random.shuffle(train_index)
        train_loader = DataLoader(train_datasets[train_index], shuffle=False, batch_size=batch_size)
        model.train()
        train_loss = 0
        start = 0
        for data in train_loader:
            data.x = torch.ones_like(data.x)
            data.edge_attr = torch.ones_like(data.edge_attr)
            data = data.to(device)
            optimizer.zero_grad()
            if use_hodge:
                end = start + train_loader.batch_size if start + train_loader.batch_size <= len(train_hodge) else len(train_hodge)
                loss = criterion(model(data, [train_hodge[train_index[ti]] for ti in range(start, end)]).view(-1), data.y)
            else:
                loss = criterion(model(data).view(-1), data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step(loss)
            train_loss += (loss.detach().cpu()) * data.num_graphs
            start += train_loader.batch_size
        train_loss /= len(train_datasets)

        model.eval()
        start = 0
        with torch.no_grad():
            test_score = torch.FloatTensor([0])
            for data in test_loader:
                data.x = torch.ones_like(data.x)
                data.edge_attr = torch.ones_like(data.edge_attr)
                data = data.to(device)
                if use_hodge:
                    end = start + test_loader.batch_size if start + test_loader.batch_size <= len(test_hodge) else len(
                        test_hodge)
                    loss = test_criterion(model(data, [test_hodge[ti] for ti in range(start, end)]).view(-1),
                                  data.y).detach().cpu()
                else:
                    loss = test_criterion(model(data).view(-1), data.y).detach().cpu()
                test_score += loss * data.num_graphs
                start += test_loader.batch_size
            test_score /= len(test_datasets)
            best_test_score = test_score if best_test_score > test_score else best_test_score
        print("Seed:{}, Epoch:{}, Training MSE:{}, Test MAE: {}, Best MAE: {}".format(seed, i, train_loss, test_score, best_test_score))
        with open(res_dir + "/log.txt", "a") as f:
            f.write("Seed:{}, Epoch:{}, Training MSE:{}, Test MAE: {}, Best MAE: {}\n".format(seed, i, train_loss, test_score, best_test_score))
    del model; del optimizer; del scheduler
    curves.append(best_test_score.item())
    t2 = time.time()
    log = "Seed:{}, Best Test MAE: {}, Time per epoch: {}\n".format(seed, best_test_score, (t2 - t1) / epochs)
    print(log)
    with open(res_dir + "/log.txt", "a") as f:
        f.write(log)
    average_time +=  ((t2 - t1) / epochs)
log = "Mean MAE: {}, std: {}, Mean Time per epoch: {}\n".format(np.mean(curves), np.std(curves), average_time / 5)
print(log)
with open(res_dir + "/log.txt", "a") as f:
    f.write(log)


