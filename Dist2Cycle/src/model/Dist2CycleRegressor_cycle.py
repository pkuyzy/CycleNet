import sys
sys.path.append('..')

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.LtpConv_cycle import LtpConv
from src.model.ign import IGN2to1





class Dist2CycleRegressor(nn.Module):
    def __init__(self,
                 in_feats,
                 n_layers,
                 out_feats,
                 hidden_feats=None,
                 aggregator_type='sum',
                 weighted_edges=True,
                 fc_bias=True,
                 norm=None,
                 fc_activation=None,
                 out_activation=None,
                 initialization='kaiming',
                 verbose=False, use_hodge = False):
        super().__init__()
        
        self.verbose=verbose
        
        self.out_activation=out_activation
        
        if hidden_feats is None:
            hidden_feats=in_feats
            
        if isinstance(hidden_feats, list):
            n_layers=len(hidden_feats)+1
        elif isinstance(hidden_feats, int):
            hidden_feats=[hidden_feats]*n_layers
        
        self.use_hodge = use_hodge
        if use_hodge:
            self.ign = IGN2to1(1, in_feats, in_feats, num_layers=3, use_bn=False)
            
        if n_layers>1:
            #first layer
            convSeq=[LtpConv(in_feats, hidden_feats[0], 
                                           aggregator_type=aggregator_type, 
                                           weighted_edges=weighted_edges,
                                           fc_bias=fc_bias,
                                           norm=norm, 
                                           fc_activation=fc_activation,
                                           initialization=initialization)]
            
            
            
            #middle layers
            for l in range(1,n_layers-1):
                convSeq.append(LtpConv(hidden_feats[l-1], hidden_feats[l], 
                                           aggregator_type=aggregator_type, 
                                           weighted_edges=weighted_edges,
                                           fc_bias=fc_bias,
                                           norm=norm, 
                                           fc_activation=fc_activation,
                                           initialization=initialization))
            #final layer
            convSeq.append(LtpConv(hidden_feats[-1], out_feats, 
                                           aggregator_type=aggregator_type, 
                                           weighted_edges=weighted_edges,
                                           fc_bias=fc_bias,
                                           norm=norm, 
                                           fc_activation=fc_activation,
                                           initialization=initialization))
            
        else:
            convSeq=[LtpConv(in_feats, out_feats, 
                                           aggregator_type=aggregator_type, 
                                           weighted_edges=weighted_edges,
                                           fc_bias=fc_bias,
                                           norm=norm, 
                                           fc_activation=fc_activation,
                                           initialization=initialization)]
        self.layers=nn.ModuleList(convSeq)
        
        
    def forward(self, g, feats):
        h=feats
        
        if self.use_hodge:
            num_nodes = g.batch_num_nodes()
            start = 0
            Eigenfeat = []
            for nn in num_nodes:
                eigenval = feats[start, int((feats.size(1)-3)/2):feats.size(1)-3]
                eigenvec = feats[start:start + nn.item(), :int((feats.size(1)-3)/2)]
                if (torch.abs(eigenval - 1)<1e-4).sum() > 0:
                    eigenvec = eigenvec[:, torch.abs(eigenval - 1)<1e-4] @ eigenvec[:, torch.abs(eigenval - 1)<1e-4].T
                    Eigenfeat.append(self.ign(eigenvec.unsqueeze(0).unsqueeze(0)).transpose(1, 2).squeeze())
                else:
                    Eigenfeat.append(torch.zeros_like(feats[start:start + nn.item(), :]))
                start += nn.item()
            Eigenfeat = torch.cat(Eigenfeat, dim = 0)
            feats += Eigenfeat
                
        
        if self.verbose:
            with torch.no_grad():
                print(f'==INPUT ===({h.cpu().min(), h.cpu().mean(), h.cpu().max()})')
                print(h)
        
        for i,layer in enumerate(self.layers):
            h=layer(g,h)
            
            if self.verbose:
                with torch.no_grad():
                    print(f'==Layer {i} output ===({h.cpu().min(), h.cpu().mean(), h.cpu().max()})')
                    print(h)
        
        
        if self.out_activation is None:
            h=h
        else:
            h=self.out_activation(h)
        
        if self.verbose:
            with torch.no_grad():
                print(f'==OUTPUT ===({h.cpu().min(), h.cpu().mean(), h.cpu().max()})')
                print(h)
        
        return h
        
        
    def reset_weights(self):
        '''
        Try resetting model weights to avoid
        weight leakage.
        '''
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()