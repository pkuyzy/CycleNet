from SRDataset import SRDataset
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
from compute_hodge import compute_shortest_cycle_basis, hodge_positional_encoding
import time
from shutil import copy, rmtree
from ppgn.ppgn import PPGN
import torch.nn as nn

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_root = "./data/sr25"
hidden_dim = 128
device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
model_name = 'CycleNet_Hodge' # can be GNN/SignNet/CycleNet_Hodge/CycleNet
use_hodge = True if model_name == 'CycleNet' or model_name == 'CycleNet_edge' or model_name == 'CycleNet_Hodge' or model_name == 'CycleNet_Hodge_edge' else False
split = 2 # the hardest example

save_appendix = time.strftime("%Y%m%d%H%M%S")
res_dir = 'results/SR/' + save_appendix
print('Results will be saved in ' + res_dir)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# Backup python files.
copy('main_sr.py', res_dir)
copy('sign_net/model.py', res_dir)
copy('sign_net/cyclenet.py', res_dir)
copy('ppgn/ppgn.py', res_dir)

transform = EVDTransform('sym')
if model_name == 'GNN':
    model = GNN(1, 1, hidden_dim, hidden_dim, nlayer=5, gnn_type='GINEConv', pooling='mean')
#elif model_name == 'CycleNet':
#    model = CycleNet(1, 1 + hidden_dim, n_hid=hidden_dim, n_out=hidden_dim, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
    # model = CycleNet(2, 1, n_hid=hidden_dim, n_out=1, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'CycleNet':
    model = CycleNet_edge(1, hidden_dim, n_hid=hidden_dim, n_out=hidden_dim, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'CycleNet_Hodge':
    model = CycleNet_edge(1, hidden_dim, n_hid=hidden_dim, n_out=hidden_dim, nl_gnn=5, gnn_type='GINEConv', pooling='mean')
elif model_name == 'SignNet':
    model = SignNetGNN(1, 1, n_hid=hidden_dim, n_out=hidden_dim, nl_signnet=5, nl_gnn=5, nl_rho=8, ignore_eigval=False,
                       gnn_type='GINEConv')
elif model_name == 'PPGN':
    model = PPGN(1, hidden_dim, 5, hidden_dim)
#elif model_name == 'CycleNet_Hodge':
#    model = CycleNet_Hodge(1, 1 + hidden_dim, n_hid=hidden_dim, n_out=hidden_dim, nl_gnn=5, gnn_type='GINEConv',
#                           pooling='mean')
else:
    raise NotImplementedError
model = model.to(device)


for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


total_wrong = 0
total_num = 0
total_time = []
if True:
    dataset = SRDataset(root=data_root, transform=transform, split = split)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    if use_hodge:
        hodge_file = data_root + "/hodge_" + str(split) + ".pt"
        if not os.path.exists(hodge_file):
            print("computing hodge")
            hodge = [hodge_positional_encoding(data) for data in dataset]
            torch.save(hodge, hodge_file)
        else:
            hodge = torch.load(hodge_file)
    else:
        hodge = None

    t1 = time.time()
    model.eval()
    with torch.no_grad():
        Pred = []
        for cd, data in enumerate(dataloader):
            data = data.to(device)
            if use_hodge:
                pred = model(data, [hodge[cd]])
            else:
                pred = model(data)
            Pred.append(pred)
        Pred = torch.cat(Pred, dim = 0)
        mm = torch.pdist(Pred, p=2)
        wrong = (mm < 1e-2).sum().item()
        total_wrong += wrong
        total_num += mm.shape[0]
        test_score = 1 - (wrong / mm.shape[0])
    t2 = time.time()
    final_log = "Acc: {}, Time per epoch: {}".format(test_score, t2 - t1)
    print(final_log)
    with open(res_dir + "/log.txt", "a") as f:
        f.write(final_log)
        f.write("\n")
    total_time.append(t2 - t1)




