import torch
from utils import ensure_input_is_tensor
from constants import DEVICE


class SimplicialComplex:

    def __init__(self, X0, X1, X2, L0, L1, L2, label, val0=None, val1=None, val2=None, vec0=None, vec1=None, vec2=None, L1_cycle=None, batch=None):
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2

        # L0, L1 and L2 can either be sparse or dense but since python doesn't really have overloading, doing this instead
        self.L0 = ensure_input_is_tensor(L0)
        if L1 is not None:
            self.L1 = ensure_input_is_tensor(L1)
        else:
            self.L1 = L1
        if L2 is not None:
            self.L2 = ensure_input_is_tensor(L2)
        else:
            self.L2 = L2
        self.val2 = val2
        self.vec2 = vec2            
        self.label = label
        self.batch = batch
        self.val0 = val0
        self.vec0 = vec0
        self.val1 = val1
        self.vec1 = vec1
        self.L1_cycle = L1_cycle

    def __eq__(self, other):
        x0 = torch.allclose(self.X0, other.X0, atol=1e-5)
        x1 = torch.allclose(self.X1, other.X1, atol=1e-5)
        x2 = torch.allclose(self.X2, other.X2, atol=1e-5)
        l0 = torch.allclose(self.L0, other.L0, atol=1e-5)
        l1 = torch.allclose(self.L1, other.L1, atol=1e-5)
        l2 = torch.allclose(self.L2, other.L2, atol=1e-5)
        label = torch.allclose(self.label, other.label, atol=1e-5)
        return all([x0, x1, x2, l0, l1, l2, label])

    def unpack_features(self):
        return self.X0, self.X1, self.X2

    def unpack_laplacians(self):
        return self.L0, self.L1, self.L2
    
    def unpack_eigen(self):
        return self.val0, self.val1, self.val2, self.vec0, self.vec1, self.vec2, self.L1_cycle

    def unpack_batch(self):
        return self.batch

    def to_device(self):
        self.X0 = self.X0.to(DEVICE)
        self.X1 = self.X1.to(DEVICE)
        self.X2 = self.X2.to(DEVICE)

        self.L0 = self.L0.to(DEVICE)
        if self.L1 is not None:
            self.L1 = self.L1.to(DEVICE)
        if self.L2 is not None:
            self.L2 = self.L2.to(DEVICE)
        if self.val0 is not None:
            self.val0.to(DEVICE)
        if self.val1 is not None:
            self.val1.to(DEVICE)
        if self.val2 is not None:
            self.val2.to(DEVICE)
        if self.vec0 is not None:
            self.vec0.to(DEVICE)
        if self.vec1 is not None:
            self.vec1.to(DEVICE)
        if self.vec2 is not None:
            self.vec2.to(DEVICE)
        if self.L1_cycle is not None:
            self.L1_cycle = [cycle[0].to(DEVICE) for cycle in self.L1_cycle]

        self.batch = [batch.to(DEVICE) for batch in self.batch]
