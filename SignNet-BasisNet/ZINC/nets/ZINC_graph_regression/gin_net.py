import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 
from dgl.nn.pytorch import GINConv
"""
    GIN (no edge features)
    
"""
from layers.mlp_readout_layer import MLPReadout
from layers.mlp import MLP
from .sign_inv_net import get_sign_inv_net
from layers.ign import IGN2to1

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

class GINNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        print('BATCH NORM IN NET', self.batch_norm)
        self.residual = net_params['residual']
        print('RESIDUAL IN NET', self.residual)
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.lap_method = net_params['lap_method']
        self.lap_lspe = net_params['lap_lspe']
        print('LAP LSPE', self.lap_lspe)
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if self.pe_init == 'rand_walk' or self.lap_lspe:
            # LSPE
            self.layers = nn.ModuleList([ GINConv(MLP(hidden_dim, hidden_dim, hidden_dim, 2, use_bn=self.batch_norm, dropout=dropout, activation='relu'), 'sum') for _ in range(self.n_layers-1) ]) 
            self.layers.append(GINConv(MLP(hidden_dim, hidden_dim, out_dim, 2, use_bn=self.batch_norm, dropout=dropout, activation='relu'), 'sum'))
        else: 
            # NoPE or LapPE
            self.layers = nn.ModuleList([ GINConv(MLP(hidden_dim, hidden_dim, hidden_dim, 2, use_bn=self.batch_norm, dropout=dropout, activation='relu'), 'sum') for _ in range(self.n_layers-1) ]) 
            self.layers.append(GINConv(MLP(hidden_dim, hidden_dim, out_dim, 2, use_bn=self.batch_norm, dropout=dropout, activation='relu'), 'sum'))
        
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

        if self.pe_init == 'rand_walk' or self.lap_lspe:
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function

        if self.lap_method == 'sign_inv' or self.lap_method == 'basis_inv':
            sign_inv_net = get_sign_inv_net(net_params)
            self.sign_inv_net = sign_inv_net

        self.use_hodge = net_params['use_hodge']
        edge_dim = hidden_dim
        if self.use_hodge == "basis":
            # self.embedding_hodge = IGN2to1(1, edge_dim, edge_dim, num_layers = 5, use_bn = False)
            self.embedding_hodge_1 = IGN2to1(1, edge_dim, edge_dim, num_layers=5, use_bn=True)
        elif self.use_hodge == "PEOI":
            # self.SCB_encoder = nn.Sequential(nn.Linear(1, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(),
            #                                nn.Linear(edge_dim, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(),
            #                                nn.Linear(edge_dim, edge_dim))
            # self.SCB_encoder1 = nn.Sequential(nn.Linear(1, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(),
            #                                 nn.Linear(edge_dim, edge_dim), nn.BatchNorm1d(edge_dim), nn.ReLU(),
            #                                 nn.Linear(edge_dim, edge_dim))

            self.SCB_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32))
            self.SCB_encoder2 = nn.Sequential(nn.Linear(32 + 1, 64), nn.ReLU(), nn.Linear(64, 64))
            self.SCB_encoder4 = nn.Sequential(nn.Linear(64, edge_dim), nn.ReLU())  # , nn.Linear(edge_dim, edge_dim))
        else:
            print("Not using Cycle information")
        
    def forward(self, g, h, p, e, snorm_n, hodge_emb):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p) 
            
        if self.pe_init == 'lap_pe' and not self.lap_lspe:
            h = h + p
            p = None
        
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)


        max_beta = 0
        max_edge = 0
        for he in hodge_emb:
            max_edge = he[1].shape[0] if max_edge < he[1].shape[0] else max_edge
            max_beta = he[2].size()[0] if max_beta < he[2].size()[0] else max_beta

        L1 = []
        Pad_Masks = []

        if self.use_hodge == "basis":
            for he in hodge_emb:
                he_cycle, he_noncycle = he[0].to(h.device), he[1].to(h.device)
                # he_cycle, he_noncycle = he[0].to(e.device), he[3].to(e.device).float()
                pad_emb, pad_mask = pad_hodge_unsqueeze(he_cycle, max_edge)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)

            L1 = torch.cat(L1, dim=0).unsqueeze(1)
            Pad_Masks = torch.cat(Pad_Masks)
            L1 = self.embedding_hodge_1(L1).transpose(1, 2).squeeze()
            L1_size = L1.size()
            L1 = L1.view(-1, L1_size[-1])
            L1 = L1[Pad_Masks]
            e += L1

        elif self.use_hodge == "PEOI":
            for he in hodge_emb:
                SCB = he[2].to(h.device).to(h.dtype).to(h.device)
                pad_emb, pad_mask = pad_SCB(SCB, max_edge, max_beta)
                L1.append(pad_emb)
                Pad_Masks.append(pad_mask)
            L1 = torch.cat(L1, dim=0).unsqueeze(3).transpose(1, 2).reshape(-1, 1)
            L1_size = L1.size()[0]
            Pad_Masks = torch.cat(Pad_Masks)
            L1_1 = L1.reshape(int(L1_size / max_beta / max_edge), max_edge, 1, 1, max_beta).expand(-1, -1, max_edge, -1, -1)
            L1_2 = L1.reshape(int(L1_size / max_beta / max_edge), 1, max_edge, 1, max_beta).expand(-1, max_edge, -1, -1, -1)
            SCB_emb = self.SCB_encoder(torch.cat((L1_1, L1_2), dim=3).transpose(3, 4)).sum(dim=2)
            SCB_emb = self.SCB_encoder2(
                torch.cat((SCB_emb, L1.view(int(L1_size / max_beta / max_edge), max_edge, max_beta, 1)), dim=-1)).sum(dim=2)
            L1 = self.SCB_encoder4(SCB_emb).view(int(L1_size / max_beta), -1)
            L1 = L1[Pad_Masks]

            e += L1

        # convnets
        for conv in self.layers:
            h = conv(g, h, e)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk' or self.lap_lspe:
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp
        
        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        self.g = g # For util; To be accessed in loss() function
        
        return self.MLP_layer(hg), g
        
    def loss(self, scores, targets):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        
        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

            loss_b = ( loss_b_1 + self.lambda_loss * loss_b_2 ) / ( self.pos_enc_dim * batch_size * n) 

            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
        else:
            loss = loss_a
        
        return loss

    
    
    
    
