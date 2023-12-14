import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree, dropout_adj, to_dense_batch, to_dense_adj
from ogb.utils.features import get_atom_feature_dims
from torch_geometric.data import Data
import math
import numpy as np
from scipy.sparse.csgraph import shortest_path
from torch_scatter import scatter, scatter_mean
from ppgn.ppgn_modules import *
from ppgn.ppgn_layer import *
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import networkx as nx

# Provably Powerful Graph Networks
class PPGN(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, input_dim, output_dim, num_rb_layers, emb_dim=300, *args,
                 **kwargs):
        super(PPGN, self).__init__()

        use_embedding = False
        use_spd = False
        if "graph_pred" in kwargs:
            self.graph_pred = kwargs["graph_pred"] # delete the final graph-level pooling
        else:
            self.graph_pred = True
        self.use_cycle = False

        self.use_embedding = use_embedding
        self.use_spd = use_spd

        #initial_dim = 1 + 3 + 9  # 9 atom features + 3 bond types + adj
        initial_dim = input_dim + 1
        #initial_dim = 11
        if self.use_spd:
            initial_dim += 1

        # ppgn modules
        num_blocks = 2
        num_fc_layers = 2
        num_tasks = output_dim

        self.ppgn_rb = torch.nn.ModuleList()
        self.ppgn_rb.append(RegularBlock(num_blocks, initial_dim, emb_dim))
        for i in range(num_rb_layers - 1):
            self.ppgn_rb.append(RegularBlock(num_blocks, emb_dim, emb_dim))

        self.ppgn_fc = torch.nn.ModuleList()
        if self.graph_pred:
            self.ppgn_fc.append(FullyConnected(emb_dim * 2, emb_dim))
        else:
            self.ppgn_fc.append(FullyConnected(emb_dim * 5, emb_dim))
        for i in range(num_fc_layers - 2):
            self.ppgn_fc.append(FullyConnected(emb_dim, emb_dim))
        self.ppgn_fc.append(FullyConnected(emb_dim, num_tasks, activation_fn=None))

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        for conv in self.ppgn_rb:
            conv.reset_parameters()
        for conv in self.ppgn_fc:
            conv.reset_parameters()

    def forward(self, data):
        if self.use_embedding:
            edge_embedding = self.bond_encoder(data.edge_attr)
            node_embedding = self.atom_encoder(data.x)
        else:
            if data.edge_attr is not None:
                edge_embedding = data.edge_attr.to(torch.float)
            else:
                edge_embedding = None
            node_embedding = data.x.to(torch.float)

        # prepare dense data
        device = data.edge_index.device
        #edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(device)
        edge_adj = torch.ones(data.edge_index.shape[1], 1).to(device)
        if edge_embedding is not None:
            edge_data = torch.cat([edge_adj, edge_embedding], 1)
        else:
            edge_data = edge_adj
        dense_edge_data = to_dense_adj(
            data.edge_index, data.batch, edge_data
        )  # |graphs| * max_nodes * max_nodes * edge_data_dim

        dense_node_data, dense_node_data_info = to_dense_batch(node_embedding, data.batch)  # |graphs| * max_nodes * d
        shape = dense_node_data.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_data = torch.empty(*shape).to(device)

        if self.use_spd:
            dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(device)

        for g in range(shape[0]):
            if self.use_spd:
                g_adj = dense_edge_data[g, :, :, 0].cpu().detach().numpy()
                g_dist_mat = torch.tensor(shortest_path(g_adj))
                g_dist_mat[torch.isinf(g_dist_mat)] = 0
                g_dist_mat /= g_dist_mat.max() + 1  # normalize
                g_dist_mat = g_dist_mat.unsqueeze(0).to(device)
                dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_data[g, :, :, i] = torch.diag(dense_node_data[g, :, i])

        if self.use_spd:
            z = torch.cat([dense_dist_mat, dense_edge_data, diag_node_data], -1)
        else:
            z = torch.cat([dense_edge_data, diag_node_data], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        for rb in self.ppgn_rb:
            z = rb(z)


        if self.graph_pred:
            #z = diag_offdiag_maxpool(z)
            z = diag_offdiag_meanpool(z)
            for fc in self.ppgn_fc:
                z = fc(z)
            return F.log_softmax(z, dim=-1)

        z = diag_offdiag(z)
        z = torch.transpose(z, 1, 2).squeeze()

        for fc in self.ppgn_fc:
            z = fc(z)


        '''
        # recover the original nodes, original version, too stupid
        previous_node = -1; cg = 0
        recovered_emb = []
        for cur_node in range(len(node_embedding) - 1):
            if data.batch[cur_node] != data.batch[cur_node + 1]:
                recovered_emb.append(z[cg, : cur_node - previous_node, :])
                cg += 1; previous_node = cur_node
        recovered_emb.append(z[-1, : cur_node - previous_node + 1, :])
        recovered_emb = torch.cat(recovered_emb, dim = 0)
        '''
        output_size = z.size()[-1]
        recovered_emb = z.view(-1, output_size)[dense_node_data_info.view(-1)]
        torch.cuda.empty_cache()
        return recovered_emb, []