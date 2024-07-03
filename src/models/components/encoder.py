import numpy as np
import torch
import torch.nn as nn
from src.models.components import gather_edges
import torch.nn.functional as F
from src.models.components.layers import SinusoidalEmbedding, GaussianFourierEmbedding


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2, 1000], max_relative_feature=32, af2_relpos=False):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range
        self.max_relative_feature = max_relative_feature
        self.af2_relpos = af2_relpos

    def _transformer_encoding(self, E_idx):
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )

        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)

        return E

    def _af2_encoding(self, E_idx, residue_index=None):
        # i-j
        if residue_index is not None:
            offset = residue_index[..., None] - residue_index[..., None, :]
            offset = torch.gather(offset, -1, E_idx)
        else:
            N_nodes = E_idx.size(1)
            ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
            offset = (E_idx.float() - ii)

        relpos = torch.clip(offset.long() + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        relpos = F.one_hot(relpos, 2 * self.max_relative_feature + 1)

        return relpos

    def forward(self, E_idx, residue_index=None):

        if self.af2_relpos:
            E = self._af2_encoding(E_idx, residue_index)
        else:
            E = self._transformer_encoding(E_idx)

        return E


class ProteinEncoder(nn.Module):
    def __init__(self,
                 node_in,
                 edge_in,
                 node_features,
                 edge_features,
                 time_embedding_type: str = "sinusoidal",
                 time_embedding_dim: int = 16,
                 num_positional_embeddings=16,
                 num_rbf=16, top_k=32, af2_relpos=True):

        """ Extract protein features """
        super(ProteinEncoder, self).__init__()

        if time_embedding_dim > 0 and time_embedding_type is not None:
            # self.node_in_ = node_in + time_embedding_dim
            # self.edge_features_ = edge_features + time_embedding_dim
            node_in_ = node_in + time_embedding_dim
            self.timestep_emb_func = self._get_timestep_embedding(time_embedding_type, time_embedding_dim)
        else:
            node_in_ = node_in
            self.timestep_emb_func = None

        self.node_embedding = nn.Linear(node_in_, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.top_k = top_k
        self.num_rbf = num_rbf

        if af2_relpos:
            num_positional_embeddings = 65

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings, af2_relpos=af2_relpos)

    def _get_timestep_embedding(self, embedding_type, embedding_dim, embedding_scale=10000):
        if embedding_type == 'sinusoidal':
            emb_func = SinusoidalEmbedding(embedding_dim, scale=embedding_scale)
        elif embedding_type == 'fourier':
            emb_func = GaussianFourierEmbedding(embedding_size=embedding_dim, scale=embedding_scale)
        else:
            raise NotImplementedError
        return emb_func

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + 2 * (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, X.shape[-2]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _impute_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        return Cb

    def _atomic_distances(self, X, E_idx):

        RBF_all = []
        for i in range(X.shape[-2]):
            for j in range(X.shape[-2]):
                RBF_all.append(self._get_rbf(X[..., i, :], X[..., j, :], E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        return RBF_all

    @staticmethod
    def _normalize(tensor: torch.Tensor, dim: int = -1):
        """
        From https://github.com/drorlab/gvp-pytorch
        """
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
        )

    def _dihedral_from_four_points(self, p0, p1, p2, p3):
        uvec_0 = p2 - p1
        uvec_1 = p0 - p1
        uvec_2 = p3 - p2

        nvec_1 = self._normalize(torch.cross(uvec_0, uvec_1, dim=-1))
        nvec_2 = self._normalize(torch.cross(uvec_0, uvec_2, dim=-1))
        sgn = torch.sign((torch.cross(uvec_1, uvec_2, dim=-1) * uvec_0).sum(-1))
        dihedral = sgn * torch.arccos((nvec_1 * nvec_2).sum(-1))
        dihedral = torch.nan_to_num(dihedral)
        return dihedral

    def _pairwise_dihedrals(self, N, Ca, C, E_idx):
        L = E_idx.shape[1]
        ir_phi = self._dihedral_from_four_points(
            C[:, :, None].repeat(1, 1, L, 1),
            N[:, None, :].repeat(1, L, 1, 1),
            Ca[:, None, :].repeat(1, L, 1, 1),
            C[:, None, :].repeat(1, L, 1, 1),
        )

        ir_psi = self._dihedral_from_four_points(
            N[:, :, None].repeat(1, 1, L, 1),
            Ca[:, :, None].repeat(1, 1, L, 1),
            C[:, :, None].repeat(1, 1, L, 1),
            N[:, None, :].repeat(1, L, 1, 1),
        )

        ir_phi = torch.gather(ir_phi, 2, E_idx)
        ir_psi = torch.gather(ir_psi, 2, E_idx)
        ir_dihed = torch.stack((ir_phi, ir_psi), dim=-1)

        return ir_dihed

    def forward(self, X, S, BB_D_sincos, SC_D_sincos, chain_indices, mask, residue_index=None, t=None):
        """ Featurize coordinates as an attributed graph """

        # Build k-Nearest Neighbors graph
        X_ca = X[:, :, 1, :]
        _, E_idx, _ = self._dist(X_ca, mask)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx, residue_index)

        # Pairwise bb atomic distances
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        Cb = self._impute_CB(N, Ca, C)
        X2 = torch.stack((N, Ca, C, O, Cb), dim=-2)
        RBF_all = self._atomic_distances(X2, E_idx)

        # node features
        Vs = []

        # One-hot encoded sequence
        Vs.append(F.one_hot(S, num_classes=21).float())

        # Sin/cos encoded backbone dihedrals
        Vs.append(BB_D_sincos.view(*X.shape[:-2], -1))
        Vs.append(SC_D_sincos.view(*X.shape[:-2], -1))

        if self.timestep_emb_func is not None:
            t_node_emb = self.timestep_emb_func(t)
            Vs.append(t_node_emb.view(*X.shape[:-2], -1))

        # edge features
        same_chain = (chain_indices[:, :, None] == chain_indices[:, None, :]).to(torch.float32)
        E_type = (torch.gather(same_chain, 2, E_idx) + 1).unsqueeze(-1)

        dihed = self._pairwise_dihedrals(N, Ca, C, E_idx)
        E = torch.cat((E_positional, RBF_all, E_type, dihed), -1)

        # Embed nodes
        V = torch.cat(Vs, dim=-1)

        h_V = self.node_embedding(V)
        h_V = self.norm_nodes(h_V)
        h_E = self.edge_embedding(E)
        h_E = self.norm_edges(h_E)

        return h_V, h_E, E_idx, X
