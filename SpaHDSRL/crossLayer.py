

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F

class CrossModalFusionLayer(torch.nn.Module):
    def __init__(self, in_dim, k_neighbors=10, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.gate = torch.nn.Linear(in_dim, 1)

    def forward(self, Z1, Z2):
        """
        Z1: (n, d)
        Z2: (n, d)
        """
        n, d = Z1.shape

        Z1n = F.normalize(Z1, p=2, dim=1)
        Z2n = F.normalize(Z2, p=2, dim=1)
        Omega = Z1n @ Z2n.t()  


        N = torch.topk(Omega, k=self.k_neighbors, dim=1).indices  # (n,k)

        omega = Omega.gather(1, N)

        w = torch.softmax(omega / self.gamma, dim=1)  # (n,k)
        neigh_agg = (w.unsqueeze(-1) * Z2[N]).sum(dim=1)  # (n,d)

        alpha = torch.sigmoid(self.gate(Z1))  # (n,1)

        E = alpha * Z1 + (1 - alpha) * neigh_agg

        return E, Omega, alpha, w, N

