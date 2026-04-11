

import copy
from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from crossLayer import CrossModalFusionLayer
import torch.nn.functional as F

EPS = 1e-15

class DeepGraphInfomaxWNN(torch.nn.Module):
    
    def __init__(
        self,
        hidden_channels: int,
        encoder: Module,
        summary: Callable,
        corruption: Callable,
    ):
        
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)


    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        pos_out = self.encoder(*args, **kwargs)
        pos_z, pos_omega, alpha1, w1, N1, alpha2, w2, N2, alpha3, w3, N3 = pos_out

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_out = self.encoder(*cor_args, **cor_kwargs)
        neg_z = neg_out[0]

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary, pos_omega, alpha1, w1, N1, alpha2, w2, N2, alpha3, w3, N3


    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        """Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """

        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value
    
    def loss(self, pos_z, neg_z, summary):
        pos_logits = self.discriminate(pos_z, summary, sigmoid=False)
        neg_logits = self.discriminate(neg_z, summary, sigmoid=False)

        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))


        return loss_pos + loss_neg



    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        multi_class: str = 'auto',
        *args,
        **kwargs,
    ) -> float:
        """Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'




class GraphEncoderWNNit(nn.Module):
    def __init__(self, nsample, in_channels1, in_channels2, hidden_channels, k_neighbors, gamma):   # gamma, k_neighbors记得添
        super(GraphEncoderWNNit, self).__init__()
        self.conv1 = GCNConv(in_channels1, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv3 = GCNConv(in_channels2, hidden_channels, cached=False)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.prelu3 = nn.PReLU(hidden_channels)
        self.prelu4 = nn.PReLU(hidden_channels)


        self.conv5 = GCNConv(in_channels1, hidden_channels, cached=False)
        self.conv6 = GCNConv(hidden_channels, hidden_channels, cached=False)  
        self.conv7 = GCNConv(in_channels2, hidden_channels, cached=False)
        self.conv8 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu5 = nn.PReLU(hidden_channels)
        self.prelu6 = nn.PReLU(hidden_channels)
        self.prelu7 = nn.PReLU(hidden_channels)
        self.prelu8 = nn.PReLU(hidden_channels)

        self.cross_modal_fusion1 = CrossModalFusionLayer(hidden_channels, k_neighbors, gamma)
        self.cross_modal_fusion2 = CrossModalFusionLayer(hidden_channels, k_neighbors, gamma)
        self.cross_modal_fusion3 = CrossModalFusionLayer(hidden_channels,k_neighbors, gamma)   

        self.mddim = hidden_channels

    def forward(self, x1, x2, spatial_edge_index, feature_omics1_edge_list, feature_omics2_edge_list):

        x1_spatial = self.conv1(x1, spatial_edge_index)
        x1_spatial = self.prelu1(x1_spatial)
        x1_spatial = self.conv2(x1_spatial, spatial_edge_index)
        x1_spatial = self.prelu2(x1_spatial)
        x1_spatial = nn.functional.normalize(x1_spatial, p=2.0, dim=1)  # spatial embedding 1

        x2_spatial = self.conv3(x2, spatial_edge_index)
        x2_spatial = self.prelu3(x2_spatial)
        x2_spatial = self.conv4(x2_spatial, spatial_edge_index)
        x2_spatial = self.prelu4(x2_spatial)
        x2_spatial = nn.functional.normalize(x2_spatial, p=2.0, dim=1)   # spatial embedding 2

        x1_feature = self.conv5(x1, feature_omics1_edge_list)
        x1_feature = self.prelu5(x1_feature)
        x1_feature = self.conv6(x1_feature, feature_omics1_edge_list)
        x1_feature = self.prelu6(x1_feature)
        x1_feature = nn.functional.normalize(x1_feature, p=2.0, dim=1)  # feature embedding 1
    
        x2_feature = self.conv7(x2, feature_omics2_edge_list)
        x2_feature = self.prelu7(x2_feature)
        x2_feature = self.conv8(x2_feature, feature_omics2_edge_list)
        x2_feature = self.prelu8(x2_feature)
        x2_feature = nn.functional.normalize(x2_feature, p=2.0, dim=1)  # feature embedding 2


        Z1, Omega1, alpha1, w1, N1 = self.cross_modal_fusion1(x1_spatial, x1_feature)
        Z2, Omega2, alpha2, w2, N2 = self.cross_modal_fusion2(x2_spatial, x2_feature)
        Z, Omega, alpha3, w3, N3 = self.cross_modal_fusion3(Z1, Z2)

        return Z, Omega, alpha1, w1, N1, alpha2, w2, N2, alpha3, w3, N3
