import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
import random
import torch_geometric
from sign_net.model import GNN
from sign_net.cyclenet import CycleNet, CycleNet_Hodge, CycleNet_edge, CycleNet_Hodge_edge
from sign_net.sign_net import SignNetGNN
from sign_net.transform import EVDTransform
from compute_hodge import compute_shortest_cycle_basis, hodge_positional_encoding, compute_
from torch_geometric.data import InMemoryDataset
import time
from shutil import copy, rmtree
from ppgn.ppgn import PPGN

#data_list = [Data(...), ..., Data(...)]
#loader = DataLoader(data_list, batch_size=32)

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k = 3 # graphs that k-WL cannot differentiate
data_dict = "./data/" + str(k)
num_graphs = 200
batch_size = 16
epochs = 50
hidden_dim = 128
device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
model_name = 'PPGN' # can be GNN/SignNet/CycleNet/CycleNet_edge/CycleNet_Hodge/CycleNet_Hodge_edge
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
copy('ppgn/ppgn.py', res_dir)

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


# the cfi graph
def generate_data():
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
    return data_list


#if not os.path.exists('processed'):
#    os.mkdir('processed')

class CFI_Data(InMemoryDataset):
    def __init__(self, root = data_dict, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 0#['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    #def download(self):
        # Download to `self.raw_dir`.
    #    return 0

    def process(self):
        data_list = generate_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



transform = EVDTransform('sym')
datasets = CFI_Data(transform=transform)
train_datasets = datasets[:int(0.8 * num_graphs)]
test_datasets = datasets[int(0.8 * num_graphs):]

train_index = [i for i in range(int(0.8 * num_graphs))]
#train_loader = DataLoader(train_datasets, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(test_datasets, shuffle = False, batch_size = batch_size)

if use_hodge:
    train_file = data_dict + "/train_hodge.pt"; test_file = data_dict + "/test_hodge.pt"
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


#model = GNN(2, 1, 128, 2 * cycle_num, nlayer=5, gnn_type='GINEConv', pooling = 'mean')
if model_name == 'GNN':
    model = GNN(1, 1, hidden_dim, 1, nlayer=5, gnn_type='GINEConv', pooling = 'mean')
elif model_name == 'CycleNet':
    model = CycleNet(1, 1 + hidden_dim, n_hid = hidden_dim, n_out = 1, nl_gnn = 5, gnn_type='GINEConv', pooling='mean')
    #model = CycleNet(2, 1, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'CycleNet_edge':
    model = CycleNet_edge(1, hidden_dim, n_hid = hidden_dim, n_out = 1, nl_gnn = 5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'CycleNet_Hodge_edge':
    model = CycleNet_edge(1, hidden_dim, n_hid = hidden_dim, n_out = 1, nl_gnn = 5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'SignNet':
    model = SignNetGNN(1, 1, n_hid=hidden_dim, n_out=1, nl_signnet=5, nl_gnn=5, nl_rho=8, ignore_eigval=False,
                       gnn_type='GINEConv')
elif model_name == 'CycleNet_Hodge':
    model = CycleNet_Hodge(1, 1 + hidden_dim, n_hid = hidden_dim, n_out = 1, nl_gnn = 5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'PPGN':
    model = PPGN(1, 1, 5, hidden_dim)
else:
    raise NotImplementedError
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10,
                                                           min_lr=1e-6)
criterion = torch.nn.BCEWithLogitsLoss()
best_test_score = -1e10

t1 = time.time()
for i in range(epochs):
    random.shuffle(train_index)
    train_loader = DataLoader(train_datasets[train_index], shuffle=False, batch_size=batch_size)
    model.train()
    train_loss = 0
    start = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        if use_hodge:
            end = start + train_loader.batch_size if start + train_loader.batch_size <= len(train_hodge) else len(train_hodge)
            loss = criterion(model(data, [train_hodge[train_index[ti]] for ti in range(start, end)]).view(-1), data.y.float())
        else:
            loss = criterion(model(data).view(-1), data.y.float())
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optimizer.step()
        scheduler.step(loss)
        train_loss += (loss.detach().cpu()) * data.num_graphs
        start += train_loader.batch_size
    train_loss /= len(train_datasets)

    model.eval()
    start = 0
    with torch.no_grad():
        test_score = 0
        for data in test_loader:
            data = data.to(device)
            if use_hodge:
                end = start + test_loader.batch_size if start + test_loader.batch_size <= len(test_hodge) else len(
                    test_hodge)
                right = ((model(data, [test_hodge[ti] for ti in range(start, end)]).view(-1) > 0.5) == data.y).sum().detach().cpu().item()
                #print(criterion(model(data, [test_hodge[ti] for ti in range(start, end)]).view(-1), data.y.float()))
            else:
                right = ((model(data).view(-1) > 0.5) == data.y).sum().detach().cpu().item()
            test_score += right
            start += test_loader.batch_size
        test_score /= len(test_datasets)
        best_test_score = test_score if best_test_score < test_score else best_test_score
    print("Epoch:{}, Training Loss:{}, Test Acc: {}, Best Acc: {}".format(i, train_loss, test_score, best_test_score))
    with open(res_dir + "/log.txt", "a") as f:
        f.write("Epoch:{}, Training Loss:{}, Test Acc: {}, Best Acc: {}\n".format(i, train_loss, test_score, best_test_score))
final_log = "Best Acc: {}, Time per epoch: {}\n".format(best_test_score, (time.time() - t1) / epochs)
print(final_log)
with open(res_dir + "/log.txt", "a") as f:
    f.write(final_log)

