import torch
import torch.nn as nn
from src.models.components import gather_nodes
from src.models.components.layers import InvariantPointMessagePassing, MPNNLayer


class MpnnNet(nn.Module):
    def __init__(
            self,
            node_features: int = 128,
            edge_features: int = 128,
            hidden_dim: int = 128,
            num_mpnn_layers: int = 3,
            n_points: int = 8,
            dropout: float = 0.1,
            act: str = "relu",
            position_scale: float = 1.0,
            use_ipmp: bool = True,
            k_neighbors: int = 32,
    ):
        super().__init__()

        # Normalization and embedding
        # self.W_seq = nn.Embedding(21, hidden_dim)

        self.use_ipmp = use_ipmp
        
        if self.use_ipmp:
            self.mpnn_layers = nn.ModuleList([
                InvariantPointMessagePassing(node_features,
                                             edge_features,
                                             hidden_dim,
                                             n_points,
                                             dropout,
                                             act=act,
                                             edge_update=True,
                                             position_scale=position_scale)
                for _ in range(num_mpnn_layers)
            ])
        else:
            self.mpnn_layers = nn.ModuleList([
                MPNNLayer(hidden_dim, hidden_dim * 2, dropout=dropout, edge_update=True, act=act, scale=k_neighbors)
                for _ in range(num_mpnn_layers)
            ])


    def forward(self, h_V, h_E, E_idx, X, S, mask):

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        if self.use_ipmp:
            for layer in self.mpnn_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)
        else:
            for layer in self.mpnn_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # h_S = self.W_seq(S)
        # h_VS = torch.cat((h_V, h_S), -1)

        return h_V



