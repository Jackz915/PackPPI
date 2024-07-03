import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import get_act_fxn, cat_neighbors_nodes
from src.utils.features import get_bb_frames


class MLP(nn.Module):
    def __init__(self, num_in, num_inter, num_out, num_layers, act='relu', bias=True):
        super().__init__()

        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)
        self.W_inter = nn.ModuleList([nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)])
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)

        # Activation function
        self.act = get_act_fxn(act)

    def forward(self, X):
        # Embed inputs with input layer
        X = self.act(self.W_in(X))

        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))

        # Get output from output layer
        X = self.W_out(X)

        return X


class InvariantPointMessagePassing(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_points=8, dropout=0.1, act='relu', edge_update=False,
                 position_scale=10.0):
        super().__init__()

        self.edge_update = edge_update
        self.n_points = n_points
        self.position_scale = position_scale
        self.points_fn_node = nn.Linear(node_dim, n_points * 3)
        if edge_update:
            self.points_fn_edge = nn.Linear(node_dim, n_points * 3)

        # Input to message is: 2*node_dim + edge_dim + 3*3*n_points
        self.node_message_fn = MLP(2 * node_dim + edge_dim + 9 * n_points, hidden_dim, hidden_dim, 3, act=act)
        if edge_update:
            self.edge_message_fn = MLP(2 * node_dim + edge_dim + 9 * n_points, hidden_dim, hidden_dim, 3, act=act)

        # Dropout and layer norms
        n_layers = 2
        if edge_update:
            n_layers = 4
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Feedforward layers
        self.node_dense = MLP(hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act)
        if edge_update:
            self.edge_dense = MLP(hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act)

    def _get_message_input(self, h_V, h_E, E_idx, X, edge=False):
        # Get backbone global frames from N, CA, and C
        bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])
        bb_to_global = bb_to_global.scale_translation(1 / self.position_scale)

        # Generate points in local frame of each node
        if not edge:
            p_local = self.points_fn_node(h_V).reshape((*h_V.shape[:-1], self.n_points, 3))  # [B, L, N, 3]
        else:
            p_local = self.points_fn_edge(h_V).reshape((*h_V.shape[:-1], self.n_points, 3))  # [B, L, N, 3]

        # Project points into global frame
        p_global = bb_to_global[..., None].apply(p_local)  # [B, L, N, 3]
        p_global_expand = p_global.unsqueeze(-3).expand(*E_idx.shape, *p_global.shape[-2:])  # [B, L, K, N, 3]

        # Get neighbor points in global frame for each node
        neighbor_idx = E_idx.view((*E_idx.shape[:-2], -1))  # [B, LK]
        neighbor_p_global = torch.gather(p_global, -3,
                                         neighbor_idx[..., None, None].expand(*neighbor_idx.shape, self.n_points, 3))
        neighbor_p_global = neighbor_p_global.view(*E_idx.shape, self.n_points, 3)  # [B, L, K, N, 3]

        # Form message components:
        # 1. Node i's local points
        p_local_expand = p_local.unsqueeze(-3).expand(*E_idx.shape, *p_local.shape[-2:])  # [B, L, K, N, 3]

        # 2. Distance between node i's local points and its CA
        p_local_norm = torch.sqrt(torch.sum(p_local_expand ** 2, dim=-1) + 1e-8)  # [B, L, K, N]

        # 3. Node j's points in i's local frame
        neighbor_p_local = bb_to_global[..., None, None].invert_apply(neighbor_p_global)  # [B, L, K, N, 3]

        # 4. Distance between node j's points in i's local frame and i's CA
        neighbor_p_local_norm = torch.sqrt(torch.sum(neighbor_p_local ** 2, dim=-1) + 1e-8)  # [B, L, K, N]

        # 5. Distance between node i's global points and node j's global points
        neighbor_p_global_norm = torch.sqrt(
            torch.sum(
                (p_global_expand - neighbor_p_global) ** 2,
                dim=-1) + 1e-8)  # [B, L, K, N]

        # Node message
        node_expand = h_V.unsqueeze(-2).expand(*E_idx.shape, h_V.shape[-1])
        neighbor_edge = cat_neighbors_nodes(h_V, h_E, E_idx)
        message_in = torch.cat(
            (node_expand,
             neighbor_edge,
             p_local_expand.view((*E_idx.shape, -1)),
             p_local_norm,
             neighbor_p_local.view((*E_idx.shape, -1)),
             neighbor_p_local_norm,
             neighbor_p_global_norm), dim=-1)

        return message_in

    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        # Get message fn input
        message_in = self._get_message_input(h_V, h_E, E_idx, X)

        # Update nodes
        node_m = self.node_message_fn(message_in)
        if mask_attend is not None:
            node_m = node_m * mask_attend[..., None]
        node_m = torch.mean(node_m, dim=-2)
        h_V = self.norm[0](h_V + self.dropout[0](node_m))
        node_m = self.node_dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](node_m))
        if mask_V is not None:
            h_V = h_V * mask_V[..., None]

        if self.edge_update:
            # Get message fn input
            message_in = self._get_message_input(h_V, h_E, E_idx, X, edge=True)

            # Update edges
            edge_m = self.edge_message_fn(message_in)
            if mask_attend is not None:
                edge_m = edge_m * mask_attend[..., None]
            h_E = self.norm[2](h_E + self.dropout[2](edge_m))
            edge_m = self.edge_dense(h_E)
            h_E = self.norm[3](h_E + self.dropout[3](edge_m))
            if mask_attend is not None:
                h_E = h_E * mask_attend[..., None]

        return h_V, h_E


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, edge_update=False, act='relu', extra_params=0):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.edge_update = edge_update

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.W_v = MLP(num_hidden + num_in, num_hidden + extra_params, num_hidden, num_layers=3, act=act)
        self.dense = MLP(num_hidden, num_hidden * 4, num_hidden, num_layers=2, act=act)

        self.act = get_act_fxn(act)

        if edge_update:
            self.W_e = MLP(num_hidden + num_in, num_hidden + extra_params, num_hidden, num_layers=3, act=act)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, E_idx=None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        if torch.is_tensor(E_idx):
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
        else:
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W_v(h_EV)
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout[0](dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if self.edge_update:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W_e(h_EV)
            h_E = self.norm2(h_E + self.dropout2(h_message))

            return h_V, h_E
        else:
            return h_V


class SigmaEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims,
                 sigma_dim, embed_type="sinusoidal", operation="pre_concat"):

        super(SigmaEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sigma_dim = sigma_dim
        self.embed_func = self.get_timestep_embedding(embed_type, sigma_dim)
        self.operation = operation
        if self.operation == "post_add":
            self.sigma_linear = nn.Linear(sigma_dim, hidden_dims)
            self.mlp = MLP(input_dim, input_dim, hidden_dims, 2)
        elif self.operation == "pre_concat":
            self.mlp = MLP(input_dim + sigma_dim, input_dim, hidden_dims, 2)

    def get_timestep_embedding(self, embedding_type, embedding_dim, embedding_scale=10000):
        if embedding_type == 'sinusoidal':
            emb_func = SinusoidalEmbedding(embedding_dim, scale=embedding_scale)
        elif embedding_type == 'fourier':
            emb_func = GaussianFourierEmbedding(embedding_size=embedding_dim, scale=embedding_scale)
        else:
            raise NotImplementedError
        return emb_func

    def forward(self, input, sigma):
        sigma_embed = self.embed_func(sigma)
        sigma_embed = sigma_embed.reshape(input.shape[0], -1, self.sigma_dim)
        if self.operation == "post_add":
            hidden = self.mlp(input)
            hidden = hidden + self.sigma_linear(sigma_embed)
            return hidden
        elif self.operation == "pre_concat":
            hidden = self.mlp(torch.cat((input, sigma_embed), dim=-1))
            return hidden


class SinusoidalEmbedding(nn.Module):
    """ Code adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """

    def __init__(self, embedding_dim, max_positions=10000, scale=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, timesteps):
        timesteps *= self.scale
        assert timesteps.ndim == 1
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb


class GaussianFourierEmbedding(nn.Module):
    """ Code adapted from https://github.com/yang-song/score_sde_pytorch/blob
    /1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32 """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x *= self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return emb





