import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
from data.generate_data_SPT import compute_SPT_bases
from data.generate_data_SL import compute_shortest_cycle_basis

# The dataset pickle and index files are in ./data/molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']




class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx], self.split, idx
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g

def cycle_positional_encoding(g, pos_enc_dim, tau=0):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    #N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    #L = sp.eye(g.number_of_nodes()) - N * A * N
    #L = L.toarray()

    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray() + tau / g.number_of_nodes()
    #N = np.diag((dgl.backend.asnumpy(g.in_degrees()).clip(1) + tau) ** -0.5)
    #L = np.eye(g.number_of_nodes()) - N @ A @ N

    Cycle_basis, _ = compute_SPT_bases(g.to_networkx().to_undirected())

    # the eigenvec version
    Cycle_A = Cycle_basis.mm(Cycle_basis.T)
    EigVal, EigVec = np.linalg.eig(Cycle_A)
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    Pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    Pos_enc = torch.cat((Pos_enc, Pos_enc), dim = 1).view(g.number_of_edges(), -1)
    g.edata['pos_enc'] = Pos_enc
    # zero padding to the end if n < pos_enc_dim
    n = int(g.number_of_edges() / 2)
    if n <= pos_enc_dim:
        g.edata['pos_enc'] = F.pad(g.edata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))

    '''
    # original cycle version
    Pos_enc = torch.cat((Cycle_basis, Cycle_basis), dim=1).view(g.number_of_edges(), -1)
    g.edata['pos_enc'] = Pos_enc[:, :pos_enc_dim]

    # zero padding to the end if n < pos_enc_dim
    n = g.edata['pos_enc'].size()[1]
    if n <= pos_enc_dim:
        g.edata['pos_enc'] = F.pad(g.edata['pos_enc'], (0, pos_enc_dim - n), value=float('0'))
    '''
    g.edata['cyclevec'] = g.edata['pos_enc']
    #print(g.edata['pos_enc'].size())

    
    return g


def hodge_positional_encoding(g):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    '''
    # hodge laplacian, version 1, uses the entire hodge lap
    B = g.incidence_matrix('both').to_dense()
    L1 = (B.T.to_sparse())@B
    maxeig = np.linalg.eigvalsh(L1)[-1]
    L1 = L1 / maxeig
    return L1.to_sparse(), L1.to_sparse()
    '''

    # version 2, separate cycles and non-cycles
    B = g.incidence_matrix('both').to_dense()
    L1 = (B.T.to_sparse()) @ B
    #maxeig = np.linalg.eigvalsh(L1)[-1]
    #L1 = L1 / maxeig


    EigVal, EigVec = np.linalg.eigh(L1)
    idx = EigVal.argsort()  # increasing order
    #print(idx)
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    #print(EigVec.T @ EigVec)
    thres = EigVal <= 1e-5
    #L1_cycle = torch.from_numpy(EigVec[:, thres]).float()
    #L1_noncycle = torch.from_numpy(EigVec[:, ~thres]).float()
    L1_cycle = torch.from_numpy(EigVec[:, thres] @ EigVec[:, thres].T).float()
    L1_noncycle = torch.from_numpy(EigVec[:, ~thres] @ EigVec[:, ~thres].T).float()
    #maxc = np.linalg.eigvalsh(L1_cycle)[-1]
    #maxnc = np.linalg.eigvalsh(L1_noncycle)[-1]
    #L1_cycle = torch.from_numpy(EigVec[:, thres]).float() / maxc
    #L1_cycle = L1_cycle / maxc
    #L1_noncycle = L1_noncycle / maxnc


    #return L1_cycle.to_sparse(), L1.to_sparse()

    # add the information of 2-cells
    SCB = compute_shortest_cycle_basis((dgl.to_networkx(g)).to_undirected())
    beta, le = SCB.shape
    SCB = np.concatenate((SCB.reshape(beta, le, 1), SCB.reshape(beta, le, 1)), axis = 2).reshape(beta, 2 * le)

    # load the information of 2-cells (shortest cycle basis)
    cell_L1 = L1 + SCB.T @ SCB

    maxeigcell = np.linalg.eigvalsh(cell_L1)[-1]
    if maxeigcell > 1e-5:
        cell_L1 = cell_L1 / maxeigcell


    maxeig = np.linalg.eigvalsh(L1)[-1]
    if maxeig > 1e-5:
        L1 = L1 / maxeig

    return L1_cycle.to_sparse(), L1.to_sparse(), torch.from_numpy(SCB).to_sparse(), cell_L1.float().to_sparse()#, torch.from_numpy(EigVec[:, thres]).float()#L1_noncycle.to_sparse()


def lap_positional_encoding(g, pos_enc_dim, tau=0):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    #N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    #L = sp.eye(g.number_of_nodes()) - N * A * N
    #L = L.toarray()

    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray() + tau / g.number_of_nodes()
    N = np.diag((dgl.backend.asnumpy(g.in_degrees()).clip(1) + tau) ** -0.5)
    L = np.eye(g.number_of_nodes()) - N @ A @ N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    Eigval = torch.from_numpy(EigVal[1:pos_enc_dim + 1]).float()
    Eigval = torch.round(Eigval * 10 ** 5) / (10 ** 5)
    g.ndata['eigval'] = Eigval.unsqueeze(0).repeat(EigVec.shape[0], 1)

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))
        g.ndata['eigval'] = F.pad(g.ndata['eigval'], (0, pos_enc_dim - n + 1), value=float('0'))

    g.ndata['eigvec'] = g.ndata['pos_enc']
    
    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


def make_full_graph(g, adaptive_weighting=None):

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']
    
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass
    
    try:
        full_g.ndata['eigvec'] = g.ndata['eigvec']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 

    
    # This code section only apply for GraphiT --------------------------------------------
    if adaptive_weighting is not None:
        p_steps, gamma = adaptive_weighting
    
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        
        # Adaptive weighting k_ij for each edge
        if p_steps == "qtr_num_nodes":
            p_steps = int(0.25*n)
        elif p_steps == "half_num_nodes":
            p_steps = int(0.5*n)
        elif p_steps == "num_nodes":
            p_steps = int(n)
        elif p_steps == "twice_num_nodes":
            p_steps = int(2*n)

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = torch.from_numpy(k_RW_power.toarray())

        # Assigning edge features k_RW_eij for adaptive weighting during attention
        full_edge_u, full_edge_v = full_g.edges()
        num_edges = full_g.number_of_edges()

        k_RW_e_ij = []
        for edge in range(num_edges):
            k_RW_e_ij.append(k_RW_power[full_edge_u[edge], full_edge_v[edge]])

        full_g.edata['k_RW'] = torch.stack(k_RW_e_ij,dim=-1).unsqueeze(-1).float()
    # --------------------------------------------------------------------------------------
        
    return full_g


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading ZINC datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs], to get all graphs with the same nodes
    def collate_basis(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, splits, idx = map(list, zip(*samples))
        labels = (torch.cat(labels)).unsqueeze(1)
        # labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        batched_graph = dgl.batch(graphs)

        # encode information for basis inv
        Projs = []
        for graph in graphs:
            N = graph.number_of_nodes()
            uniq_vals, inv_inds, counts = graph.ndata['eigval'][0].unique(return_inverse=True, return_counts=True)
            uniq_mults = counts.unique()
            eigenspaces = torch.tensor_split(graph.ndata['eigvec'],torch.cumsum(counts,0).cpu(),dim=1)[:-1]
            projectors = [V @ V.T for V in eigenspaces]
            projectors = [P.reshape(1,1,N,N) for P in projectors]
            same_size_projs = {mult.item():[] for mult in uniq_mults}
            for i in range(len(projectors)):
                mult = counts[i].item()
                same_size_projs[mult].append(projectors[i])
            for mult, projs in same_size_projs.items():
                same_size_projs[mult]  = torch.cat(projs, dim = 0)
            Projs.append(same_size_projs)

        # encode batch information for IGN
        batch_info = []
        for cnt_batch, g in enumerate(graphs):
            for _ in range(g.number_of_nodes()):
                batch_info.append(cnt_batch)
        batch_info = torch.LongTensor(batch_info)
        return batched_graph, labels, snorm_n, Projs, batch_info, None


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        #graphs, labels = map(list, zip(*samples))
        graphs, labels, splits, idx = map(list, zip(*samples))
        labels = (torch.cat(labels)).unsqueeze(1)
        #labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)

        # encode batch information for IGN
        batch_info = []
        for cnt_batch, g in enumerate(graphs):
            for _ in range(g.number_of_nodes()):
                batch_info.append(cnt_batch)
        batch_info = torch.LongTensor(batch_info)

        # encode hodge information
        if hasattr(getattr(self, splits[0]), "hodge_emb"):
            # version 1, original hodge laplacian
            #hodge_emb = [getattr(self, splits[0]).hodge_emb[id].to_dense() for id in idx]
            # version 2, cycle and non-cycle
            #hodge_emb = [((getattr(self, splits[0]).hodge_emb[id])[0].to_dense(), (getattr(self, splits[0]).hodge_emb[id])[1].to_dense()) for id in idx]
            # add SCB and cell information
            hodge_emb = [((getattr(self, splits[0]).hodge_emb[id])[0].to_dense(),
                          (getattr(self, splits[0]).hodge_emb[id])[1].to_dense(),
                          (getattr(self, splits[0]).hodge_emb[id])[2].to_dense(),
                          (getattr(self, splits[0]).hodge_emb[id])[3].to_dense(),
                          #(getattr(self, splits[0]).hodge_emb[id])[4],
                          ) for id in idx]
        else:
            print("None")
            hodge_emb = None
        return batched_graph, labels, snorm_n, None, batch_info, hodge_emb

    def _add_cycle_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [cycle_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [cycle_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [cycle_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_hodge_encodings(self):
        #torch.save(torch.zeros(1), "./hodge/1.pt")
        #self.train.hodge_emb = [hodge_positional_encoding(g) for g in self.train.graph_lists]
        #self.val.hodge_emb = [hodge_positional_encoding(g) for g in self.val.graph_lists]
        #self.test.hodge_emb = [hodge_positional_encoding(g) for g in self.test.graph_lists]

        if os.path.exists("./hodge/train.pt"):
            self.train.hodge_emb = torch.load("./hodge/train.pt")
        else:
            self.train.hodge_emb = [hodge_positional_encoding(g) for g in self.train.graph_lists]
            torch.save(self.train.hodge_emb, "./hodge/train.pt")
        if os.path.exists("./hodge/val.pt"):
            self.val.hodge_emb = torch.load("./hodge/val.pt")
        else:
            self.val.hodge_emb = [hodge_positional_encoding(g) for g in self.val.graph_lists]
            torch.save(self.val.hodge_emb, "./hodge/val.pt")
        if os.path.exists("./hodge/test.pt"):
            self.test.hodge_emb = torch.load("./hodge/test.pt")
        else:
            self.test.hodge_emb = [hodge_positional_encoding(g) for g in self.test.graph_lists]
            torch.save(self.test.hodge_emb, "./hodge/test.pt")


    def _add_lap_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]
    
    def _add_eig_vecs(self, pos_enc_dim):

        # This is used if we visualize the eigvecs
        self.train.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.test.graph_lists]
    
    def _init_positional_encodings(self, pos_enc_dim, type_init):
        
        # Initializing positional encoding randomly with l2-norm 1
        self.train.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.train.graph_lists]
        self.val.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.val.graph_lists]
        self.test.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.test.graph_lists]
        
    def _make_full_graph(self, adaptive_weighting=None):
        self.train.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.test.graph_lists]




