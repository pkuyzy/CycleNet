import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from sign_net.model import GNN, sort_edge_index, GNN_edge
from sign_net.ign import IGN2to1
from sign_net.sign_net import SignNet

def pad_hodge_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    x_mask = torch.zeros(padlen).to(x.device); x_mask = x_mask.bool()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x_mask[:xlen] = 1
        x = new_x
    else:
        x_mask[:] = 1
    return x.unsqueeze(0), x_mask

def pad_SCB(x, edgelen, betalen):
    xsize = x.size()
    x_mask = torch.zeros(edgelen).to(x.device); x_mask = x_mask.bool()
    if xsize[1] < edgelen or xsize[0] < betalen:
        new_x = x.new_zeros([betalen, edgelen], dtype=x.dtype)
        new_x[:xsize[0], :xsize[1]] = x
        x_mask[:xsize[1]] = 1
        x = new_x
    else:
        x_mask[:] = 1
    return x.unsqueeze(0), x_mask

class CycleNet(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_gnn, gnn_type='GINEConv', pooling='mean'):
        super().__init__()

        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)
        #self.gnn = GNN(node_feat, edge_feat, n_hid, 128, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)

        #self.sign_net = SignNet(n_hid, nl_gnn, nl_rho=4, ignore_eigval=False)
        edge_dim = n_hid
        #self.embedding_hodge = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=False)
        #self.embedding_hodge_1 = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=True)

        #self.SCB_encoder = nn.Sequential(nn.Linear(1, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64))
        #self.SCB_encoder2 = nn.Sequential(nn.Linear(64 + 1, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128))
        #self.SCB_encoder4 = nn.Sequential(nn.Linear(128, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        #self.final_mlp = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, n_out))

        self.SCB_encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        self.SCB_encoder2 = nn.Sequential(nn.Linear(64 + 1, 128), nn.ReLU(), nn.Linear(128, 128))
        self.SCB_encoder4 = nn.Sequential(nn.Linear(128, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        #self.final_mlp = nn.Sequential(nn.Linear(128 + 64, 128), nn.ReLU(), nn.Linear(128, n_out))
    #def reset_parameters(self):
    #    self.gnn.reset_parameters()

    def forward(self, data, hodge_emb = None):
        if hodge_emb is not None:

            max_beta = 0
            max_edge = 0
            for he in hodge_emb:
                max_edge = he[1].shape[0] if max_edge < he[1].shape[0] else max_edge
                max_beta = he[2].size()[0] if max_beta < he[2].size()[0] else max_beta
                L1 = []
                Pad_Masks = []
            for he in hodge_emb:
                SCB = torch.abs(he[2].to_dense()).to(data.x.dtype).to(data.x.device)
                pad_emb, pad_mask = pad_SCB(SCB, max_edge, max_beta)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)


            Pad_Masks = torch.cat(Pad_Masks)
            L1 = torch.cat(L1, dim=0).unsqueeze(3).transpose(1, 2)
            L1_size = L1.size()
            SCB_emb = self.SCB_encoder(L1.sum(dim = 1)).unsqueeze(dim = 1).expand(-1, L1_size[1], -1, -1)
            SCB_emb = self.SCB_encoder2(torch.cat((SCB_emb, L1), dim=-1)).sum(dim = 2)
            L1 = self.SCB_encoder4(SCB_emb).reshape(L1_size[0] * L1_size[1], -1)
            L1 = L1[Pad_Masks]


        else:
            L1 = None

        # original version, use GNN to transfer positional encoding
        return self.gnn(data, None, L1)

        #second version, directly sum GNN and the embedding


        #return self.final_mlp(torch.cat((self.gnn(data), L1.sum(dim = 1)), dim = -1))
        #return self.final_mlp(torch.cat((self.gnn(data), L1), dim=-1))


class CycleNet_Hodge(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_gnn, gnn_type='GINEConv', pooling='mean'):
        super().__init__()

        # self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)

        # self.sign_net = SignNet(n_hid, nl_gnn, nl_rho=4, ignore_eigval=False)
        edge_dim = n_hid
        #self.embedding_hodge = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=False)
        self.embedding_hodge_1 = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=True)
        #self.embedding_hodge_1 = IGN2to1(1, edge_dim, 1, num_layers=5, use_bn=False)

    # def reset_parameters(self):
    #    self.gnn.reset_parameters()

    def forward(self, data, hodge_emb=None):
        if hodge_emb is not None:

            max_beta = 0
            max_edge = 0
            for he in hodge_emb:
                max_edge = he[1].shape[0] if max_edge < he[1].shape[0] else max_edge
                max_beta = he[2].size()[0] if max_beta < he[2].size()[0] else max_beta
                L1 = []
                Pad_Masks = []


            for he in hodge_emb:
                #he_cycle, he_noncycle = he[0].to(data.x.device), he[1].to_dense().to(data.x.dtype).to(data.x.device)
                he_cycle, he_noncycle = he[0].to(data.x.device), he[3].to_dense().to(data.x.dtype).to(data.x.device)
                pad_emb, pad_mask = pad_hodge_unsqueeze(he_noncycle, max_edge)
                #pad_emb, pad_mask = pad_hodge_unsqueeze(he_cycle, max_edge)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)

            L1 = torch.cat(L1, dim=0).unsqueeze(1)
            Pad_Masks = torch.cat(Pad_Masks)
            L1 = self.embedding_hodge_1(L1).transpose(1, 2).squeeze()
            L1_size = L1.size()
            L1 = L1.view(-1, L1_size[-1])
            L1 = L1[Pad_Masks]



        else:
            L1 = None

        # original version, use GNN to transfer positional encoding
        return self.gnn(data, None, L1)
        #return L1.sum(dim = 1)



class CycleNet_edge(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_gnn, gnn_type='GINEConv', pooling='mean'):
        super().__init__()

        self.gnn = GNN_edge(edge_feat, 1, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)
        #self.gnn = GNN(node_feat, edge_feat, n_hid, 128, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)

        #self.sign_net = SignNet(n_hid, nl_gnn, nl_rho=4, ignore_eigval=False)
        edge_dim = n_hid
        #self.embedding_hodge = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=False)
        #self.embedding_hodge_1 = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=True)

        #self.SCB_encoder = nn.Sequential(nn.Linear(1, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64))
        #self.SCB_encoder2 = nn.Sequential(nn.Linear(64 + 1, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128))
        #self.SCB_encoder4 = nn.Sequential(nn.Linear(128, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        #self.final_mlp = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, n_out))

        #self.SCB_encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.SCB_encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        self.SCB_encoder2 = nn.Sequential(nn.Linear(64 + 1, 128), nn.ReLU(), nn.Linear(128, 128))
        self.SCB_encoder4 = nn.Sequential(nn.Linear(128, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        #self.final_mlp = nn.Sequential(nn.Linear(128 + 64, 128), nn.ReLU(), nn.Linear(128, n_out))
    #def reset_parameters(self):
    #    self.gnn.reset_parameters()

    def forward(self, data, hodge_emb = None):
        if hodge_emb is not None:

            max_beta = 0
            max_edge = 0
            for he in hodge_emb:
                max_edge = he[1].shape[0] if max_edge < he[1].shape[0] else max_edge
                max_beta = he[2].size()[0] if max_beta < he[2].size()[0] else max_beta
                L1 = []
                Pad_Masks = []
            for he in hodge_emb:
                SCB = torch.abs(he[2].to_dense()).to(data.x.dtype).to(data.x.device)
                pad_emb, pad_mask = pad_SCB(SCB, max_edge, max_beta)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)


            Pad_Masks = torch.cat(Pad_Masks)
            L1 = torch.cat(L1, dim=0).unsqueeze(3).transpose(1, 2)
            L1_size = L1.size()

            #right = (L1.sum(dim=1) > 10).float()
            #L1 = self.SCB_encoder((L1.sum(dim=1) > 10).float())

            SCB_emb = self.SCB_encoder(L1.sum(dim = 1)).unsqueeze(dim = 1).expand(-1, L1_size[1], -1, -1)
            SCB_emb = self.SCB_encoder2(torch.cat((SCB_emb, L1), dim=-1)).sum(dim = 2)
            L1 = self.SCB_encoder4(SCB_emb).reshape(L1_size[0] * L1_size[1], -1)
            L1 = L1[Pad_Masks]



        else:
            L1 = None

        new_edge_index = torch.cat((torch.arange(data.edge_index.size()[1]).unsqueeze(1).to(
            data.edge_index.device), data.edge_index[0].unsqueeze(1)), dim=1).T
        new_edge_index1 = torch.cat((torch.arange(data.edge_index.size()[1]).unsqueeze(1).to(
            data.edge_index.device), data.edge_index[1].unsqueeze(1)), dim=1).T
        new_edge_index = torch.cat((new_edge_index, new_edge_index1), dim = 1)
        new_edge_attr = torch.ones(new_edge_index.size(1), device=new_edge_index.device)
        B = torch.sparse_coo_tensor(new_edge_index, new_edge_attr, (data.edge_index.size(1), data.x.size(0)),
                                device=data.edge_index.device)
        new_edge_index = torch.nonzero((B@(B.to_dense().T))).T


        #return L1.sum(dim = 1)
        #return scatter(L1, data.batch[data.edge_index[0]], dim=0, reduce='mean')
        return self.gnn(L1, new_edge_index, data.batch[data.edge_index[0]])

class CycleNet_Hodge_edge(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_gnn, gnn_type='GINEConv', pooling='mean'):
        super().__init__()

        self.gnn = GNN_edge(edge_feat, 1, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)
        #self.gnn = GNN(node_feat, edge_feat, n_hid, 128, nlayer=nl_gnn, gnn_type=gnn_type, pooling=pooling)

        #self.sign_net = SignNet(n_hid, nl_gnn, nl_rho=4, ignore_eigval=False)
        edge_dim = n_hid
        #self.embedding_hodge = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=False)
        self.embedding_hodge_1 = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=True)

    #def reset_parameters(self):
    #    self.gnn.reset_parameters()

    def forward(self, data, hodge_emb = None):
        if hodge_emb is not None:

            max_beta = 0
            max_edge = 0
            for he in hodge_emb:
                max_edge = he[1].shape[0] if max_edge < he[1].shape[0] else max_edge
                max_beta = he[2].size()[0] if max_beta < he[2].size()[0] else max_beta
                L1 = []
                Pad_Masks = []

            for he in hodge_emb:
                # he_cycle, he_noncycle = he[0].to(data.x.device), he[1].to_dense().to(data.x.dtype).to(data.x.device)
                he_cycle, he_noncycle = he[0].to(data.x.device), he[3].to_dense().to(data.x.dtype).to(data.x.device)
                pad_emb, pad_mask = pad_hodge_unsqueeze(he_noncycle, max_edge)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)

            L1 = torch.cat(L1, dim=0).unsqueeze(1)
            Pad_Masks = torch.cat(Pad_Masks)
            L1 = self.embedding_hodge_1(L1).transpose(1, 2).squeeze()
            L1_size = L1.size()
            L1 = L1.view(-1, L1_size[-1])
            L1 = L1[Pad_Masks]



        else:
            L1 = None

        new_edge_index = torch.cat((torch.arange(data.edge_index.size()[1]).unsqueeze(1).to(
            data.edge_index.device), data.edge_index[0].unsqueeze(1)), dim=1).T
        new_edge_index1 = torch.cat((torch.arange(data.edge_index.size()[1]).unsqueeze(1).to(
            data.edge_index.device), data.edge_index[1].unsqueeze(1)), dim=1).T
        new_edge_index = torch.cat((new_edge_index, new_edge_index1), dim = 1)
        new_edge_attr = torch.ones(new_edge_index.size(1), device=new_edge_index.device)
        B = torch.sparse_coo_tensor(new_edge_index, new_edge_attr, (data.edge_index.size(1), data.x.size(0)),
                                device=data.edge_index.device)
        new_edge_index = torch.nonzero((B@(B.to_dense().T))).T


        #return L1.sum(dim = 1)
        #return scatter(L1, data.batch[data.edge_index[0]], dim=0, reduce='mean')
        return self.gnn(L1, new_edge_index, data.batch[data.edge_index[0]])