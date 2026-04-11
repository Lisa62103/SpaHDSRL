

import math
import os
import torch
import random
import gudhi
import anndata
import cmcrameri
import numpy as np
import scanpy as sc
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
from model import DeepGraphInfomaxWNN, GraphEncoderWNNit





def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list


def corruptionWNNit(x1, x2, spatial_edge_index, feature_omics1_edge_list, feature_omics2_edge_list):
    # 同一个 perm 同时打乱 x1/x2（保留两模态之间的对应关系）
    perm = torch.randperm(x1.size(0), device=x1.device)
    return (
        x1[perm],
        x2[perm],
        spatial_edge_index,
        feature_omics1_edge_list,
        feature_omics2_edge_list,
    )


class SpaHDSRL(object):

    def __init__(self, adata1=None, adata2=None, count_matrix1=None, count_matrix2=None, spatial_locs=None, sample_names=None, gene_names=None):
        if adata1 and isinstance(adata1, anndata.AnnData):
            self.adata1 = adata1
        if adata2 and isinstance(adata2, anndata.AnnData):
            self.adata2 = adata2
        elif count_matrix1 is not None and count_matrix2 is not None and spatial_locs is not None:
            self.adata1 = anndata.AnnData(count_matrix1.astype(float))
            self.adata1.obsm['spatial'] = spatial_locs.astype(float)
            self.adata2 = anndata.AnnData(count_matrix2.astype(float))
            self.adata2.obsm['spatial'] = spatial_locs.astype(float)
            if gene_names:
                self.adata1.var_names = np.array(gene_names).astype(str)
                self.adata2.var_names = np.array(gene_names).astype(str)
            if sample_names:
                self.adata1.obs_names = np.array(sample_names).astype(str)
                self.adata2.obs_names = np.array(sample_names).astype(str)
        else:
            print("Please input an anndata.AnnData to initiate SpaHDSRL object.")
            exit(1)


    def preprocessing_data(self, do_norm = False, do_log = False, n_top_genes=None, do_pca = False, n_neighbors=10):
        adata1 = self.adata1
        adata2 = self.adata2

        if not adata1 or not adata2:
            print("Not enough annData objects")
            return
        if do_norm:
            sc.pp.normalize_total(adata1, target_sum=1e4)
            sc.pp.normalize_total(adata2, target_sum=1e4)
        if do_log:
            sc.pp.log1p(adata1)
            sc.pp.log1p(adata2)
        if n_top_genes:
            sc.pp.highly_variable_genes(adata1, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
            sc.pp.highly_variable_genes(adata2, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        if do_pca:
            sc.pp.pca(adata1)
            sc.pp.pca(adata2)

        spatial_locs = adata1.obsm['spatial']
        spatial_graph = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')   # spatal邻接矩阵


        adta_omics1 = adata1.X.todense() if type(adata1.X).__module__ != np.__name__ else adata1.X
        adta_omics2 = adata2.X.todense() if type(adata2.X).__module__ != np.__name__ else adata2.X
        feature_graph_omics1 = kneighbors_graph(adta_omics1, n_neighbors=n_neighbors, mode='distance')   # 这里的neighbor可以再考虑
        feature_graph_omics2 = kneighbors_graph(adta_omics2, n_neighbors=n_neighbors, mode='distance')

        self.adata1_preprocessed = adata1
        self.adata2_preprocessed = adata2
        self.feature_graph_omics1 = feature_graph_omics1
        self.feature_graph_omics2 = feature_graph_omics2
        self.spatial_graph = spatial_graph
    





    def train(self, embedding_save_filepath="./embedding.tsv", weights_save_filepath="./weights.tsv", spatial_regularization_strength=0.05, z_dim=50, lr=1e-3, wnn_epoch  = 100, total_epoch = 1000, max_patience_bef=10, max_patience_aft=30, min_stop=100, random_seed=42, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000, k_neighbors=10, gamma=0.05):
        adata1_preprocessed, adata2_preprocessed = self.adata1_preprocessed, self.adata2_preprocessed
        spatial_graph = self.spatial_graph
        feature_graph_omics1 = self.feature_graph_omics1
        feature_graph_omics2 = self.feature_graph_omics2
        if not adata1_preprocessed or not adata2_preprocessed:
            print("The data has not been preprocessed, please run preprocessing_data() method first!")
            return
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        
        model = DeepGraphInfomaxWNN(
        hidden_channels=z_dim, encoder=GraphEncoderWNNit(adata1_preprocessed.shape[0], adata1_preprocessed.shape[1], adata2_preprocessed.shape[1], z_dim, k_neighbors, gamma),
        summary=lambda z, *args, **kwargs: z.mean(dim=0),
        corruption=corruptionWNNit).to(device)
        
        expr1 = adata1_preprocessed.X.todense() if type(adata1_preprocessed.X).__module__ != np.__name__ else adata1_preprocessed.X
        expr1 = torch.tensor(expr1.copy()).float().to(device)
        
        expr2 = adata2_preprocessed.X.todense() if type(adata2_preprocessed.X).__module__ != np.__name__ else adata2_preprocessed.X
        expr2 = torch.tensor(expr2.copy()).float().to(device)
        
        spatial_edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)
        feature_omics1_edge_list = sparse_mx_to_torch_edge_list(feature_graph_omics1).to(device)
        feature_omics2_edge_list = sparse_mx_to_torch_edge_list(feature_graph_omics2).to(device)


        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_loss = np.inf
        best_params = model.state_dict()
        
        coords = torch.tensor(adata1_preprocessed.obsm['spatial']).float().to(device)

        for epoch in range(1, total_epoch):
            train_loss = 0.0

            torch.set_grad_enabled(True)
            optimizer.zero_grad()

            z, neg_z, summary, _, _, _, _, _, _, _, _, _, _ = model(expr1, expr2, spatial_edge_list, feature_omics1_edge_list, feature_omics2_edge_list)

            loss = model.loss(z, neg_z, summary)

            
            if spatial_regularization_strength > 0:
                if regularization_acceleration or adata1_preprocessed.shape[0] > 5000:
                    cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                        device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
                    z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                    c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                     cell_random_subset_2)
                    pdist = torch.nn.PairwiseDistance(p=2)

                    z_dists = pdist(z1, z2)
                    z_dists = z_dists / torch.max(z_dists)

                    sp_dists = pdist(c1, c2)
                    sp_dists = sp_dists / torch.max(sp_dists)

                    n_items = z_dists.size(dim=0)
                else:
                    z_dists = torch.cdist(z, z, p=2)
                    z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
    
                    sp_dists = torch.cdist(coords, coords, p=2)
                    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
            
                    n_items = z.size(dim=0) * z.size(dim=0)
                penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            else: penalty_1 = 0 
            
            loss = loss + spatial_regularization_strength * penalty_1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if best_loss > train_loss:
                best_loss = train_loss
                best_params = model.state_dict()

            if epoch % 10 == 1:
                print(f"Epoch {epoch}/{total_epoch}, Loss: {str(train_loss)}")

        model.load_state_dict(best_params)

        model.eval()
        with torch.no_grad():
            z, _, _, omega, alpha1, w1, N1, alpha2, w2, N2, alpha3, w3, N3 = model(expr1, expr2, spatial_edge_list, feature_omics1_edge_list, feature_omics2_edge_list)
        
        embedding = z.cpu().detach().numpy()
        crossmode_similarity = omega.cpu().detach().numpy()
        alpha1 = alpha1.squeeze(1).cpu().detach().numpy()
        alpha2 = alpha2.squeeze(1).cpu().detach().numpy()
        alpha3 = alpha3.squeeze(1).cpu().detach().numpy()   # shape: (n_spots,)

        self.embedding = embedding
        self.cross_similarity =crossmode_similarity


        self.mode1_spatial_weight = alpha1
        self.mode1_feature_weight = 1 - alpha1

        self.mode2_spatial_weight = alpha2
        self.mode2_feature_weight = 1 - alpha2

        self.mode1_weight = alpha3
        self.mode2_weight = 1 - alpha3
        self.modality_weight = pd.DataFrame({
            "mode1_weight": alpha3,
            "mode2_weight": 1 - alpha3
        }, index=adata1_preprocessed.obs_names)

        self.all_weights = pd.DataFrame({
            "mode1_spatial_weight": alpha1,
            "mode1_feature_weight": 1 - alpha1,
            "mode2_spatial_weight": alpha2,
            "mode2_feature_weight": 1 - alpha2,
            "mode1_weight": alpha3,
            "mode2_weight": 1 - alpha3
        }, index=adata1_preprocessed.obs_names)

        return embedding