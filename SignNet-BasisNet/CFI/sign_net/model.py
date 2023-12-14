import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import sign_net.model_utils.pyg_gnn_wrapper as gnn_wrapper
from sign_net.model_utils.elements import MLP, DiscreteEncoder, Identity, BN
from torch_geometric.nn.inits import reset


def sort_edge_index(
        edge_index,
        edge_attr=None,
        num_nodes=None,
        sort_by_row=True):
    num_nodes = edge_index.max() + 1

    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]

    perm = idx.argsort()

    edge_index = edge_index[:, perm]

    if edge_attr is not None:
        return edge_index, edge_attr[perm], perm
    else:
        return edge_index, perm


class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, gnn_type, dropout=0, pooling='add', bn=BN, res=True):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.edge_encoders = nn.ModuleList(
            [DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1) for _ in range(nlayer)])
        self.convs = nn.ModuleList(
            [getattr(gnn_wrapper, gnn_type)(nhid, nhid, bias=not bn) for _ in range(nlayer)])  # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False,
                                  with_norm=False if pooling == 'mean' else True)
        # self.size_embedder = nn.Embedding(200, nhid)
        self.linear = nn.Linear(2 * nhid, nhid)

        self.pooling = pooling
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        # self.size_embedder.reset_parameters()
        self.linear.reset_parameters()
        for edge_encoder, conv, norm in zip(self.edge_encoders, self.convs, self.norms):
            edge_encoder.reset_parameters()
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(self, data, additional_x=None, additional_edge=None):
        x = self.input_encoder(data.x)

        if additional_x is not None:
            x = self.linear(torch.cat([x, additional_x], dim=-1))

        ori_edge_attr = data.edge_attr
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        if additional_edge is not None:
            # mask_id = (sort_edge_index(data.edge_index, sort_by_row = False)[1] > sort_edge_index(data.edge_index, sort_by_row = True)[1])
            # large_id = torch.tensor([i for i in range(data.edge_index.size()[1])]).to(x.device)[mask_id]
            # small_id = torch.tensor([i for i in range(data.edge_index.size()[1])]).to(x.device)[~mask_id]
            # new_additional_edge = torch.empty(data.edge_index.size()[1], additional_edge.size()[1]).to(x.device)
            # new_additional_edge[large_id] = additional_edge
            # new_additional_edge[small_id] = additional_edge
            # if ori_edge_attr.ndim == 1:
            #    ori_edge_attr = ori_edge_attr.view(-1, 1)
            # ori_edge_attr = torch.cat((ori_edge_attr, new_additional_edge), dim = -1)

            if ori_edge_attr.ndim == 1:
                ori_edge_attr = ori_edge_attr.view(-1, 1)
            ori_edge_attr = torch.cat((ori_edge_attr, additional_edge), dim=-1)

        previous_x = x
        for edge_encoder, layer, norm in zip(self.edge_encoders, self.convs, self.norms):
            if ori_edge_attr.ndim > 1:
                edge_attr = edge_encoder(ori_edge_attr)
            else:
                edge_attr = edge_encoder(ori_edge_attr.float().view(-1, 1))
            x = layer(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        if self.pooling == 'mean':
            graph_size = scatter(torch.ones_like(x[:, 0], dtype=torch.int64), data.batch, dim=0, reduce='add')
            x = scatter(x, data.batch, dim=0, reduce='mean')  # + self.size_embedder(graph_size)
        elif self.pooling == 'add':
            x = scatter(x, data.batch, dim=0, reduce='add')

        x = self.output_encoder(x)
        return x


class GNN_edge(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, gnn_type, dropout=0, pooling='add', bn=BN, res=True):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.edge_encoders = nn.ModuleList(
            [DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1) for _ in range(nlayer)])
        self.convs = nn.ModuleList(
            [getattr(gnn_wrapper, gnn_type)(nhid, nhid, bias=not bn) for _ in range(nlayer)])  # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False,
                                  with_norm=False if pooling == 'mean' else True)
        # self.size_embedder = nn.Embedding(200, nhid)
        self.linear = nn.Linear(2 * nhid, nhid)

        self.pooling = pooling
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        # self.size_embedder.reset_parameters()
        self.linear.reset_parameters()
        for edge_encoder, conv, norm in zip(self.edge_encoders, self.convs, self.norms):
            edge_encoder.reset_parameters()
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(self, x, edge_index, batch, additional_x=None):
        x = self.input_encoder(x.squeeze())

        if additional_x is not None:
            x = self.linear(torch.cat([x, additional_x], dim=-1))

        ori_edge_attr = edge_index.new_zeros(edge_index.size(-1))

        previous_x = x
        for edge_encoder, layer, norm in zip(self.edge_encoders, self.convs, self.norms):
            if ori_edge_attr.ndim > 1:
                edge_attr = edge_encoder(ori_edge_attr)
            else:
                edge_attr = edge_encoder(ori_edge_attr.float().view(-1, 1))
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        if self.pooling == 'mean':
            graph_size = scatter(torch.ones_like(x[:, 0], dtype=torch.int64), batch, dim=0, reduce='add')
            x = scatter(x, batch, dim=0, reduce='mean')  # + self.size_embedder(graph_size)
        elif self.pooling == 'add':
            x = scatter(x, batch, dim=0, reduce='add')

        x = self.output_encoder(x)
        return x
