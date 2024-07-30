import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.features import get_bb_frames, torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
import src.utils.residue_constants as rc


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [...,N,C] at Neighbor indices [...,N,K] => [...,N,K,C]
    is_batched = neighbor_idx.dim() == 3
    n_feat_dims = nodes.dim() - (1 + is_batched)

    # Flatten and expand indices per batch [...,N,K] => [...,NK] => [...,NK,C]
    neighbors_flat = neighbor_idx.view((*neighbor_idx.shape[:-2], -1))
    for _ in range(n_feat_dims):
        neighbors_flat = neighbors_flat.unsqueeze(-1)
    neighbors_flat = neighbors_flat.expand(*([-1] * (1 + is_batched)), *nodes.shape[-n_feat_dims:])

    # Gather and re-pack
    neighbor_features = torch.gather(nodes, -n_feat_dims - 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape) + list(nodes.shape[-n_feat_dims:]))
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def get_act_fxn(act: str):
    if act == 'relu':
        return F.relu
    elif act == 'gelu':
        return F.gelu
    elif act == 'elu':
        return F.elu
    elif act == 'selu':
        return F.selu
    elif act == 'celu':
        return F.celu
    elif act == 'leaky_relu':
        return F.leaky_relu
    elif act == 'prelu':
        return F.prelu
    elif act == 'silu':
        return F.silu
    elif act == 'sigmoid':
        return nn.Sigmoid()


@torch.no_grad()
def get_sc_atom14_mask(S, chi_id):
    chi_atom14_mask_prev = torch.tensor(rc.sc_atom14_mask(chi_id - 1 if chi_id > 0 else 0),
                                        dtype=torch.float32,
                                        device=S.device,
                                        requires_grad=False)
    chi_atom14_mask_cur = torch.tensor(rc.sc_atom14_mask(chi_id),
                                       dtype=torch.float32,
                                       device=S.device,
                                       requires_grad=False)
    sc_atom14_mask = chi_atom14_mask_cur - chi_atom14_mask_prev

    return sc_atom14_mask[S]


# @torch.no_grad()
def get_atom14_coords(X, S, BB_D, SC_D):

    # Convert angles to sin/cos
    BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
    SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)

    # Get backbone global frames from N, CA, and C
    bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])

    # Concatenate all angles
    angle_agglo = torch.cat([BB_D_sincos, SC_D_sincos], dim=-2) # [B, L, 7, 2]

    # Get norm of angles
    norm_denom = torch.sqrt(torch.clamp(torch.sum(angle_agglo ** 2, dim=-1, keepdim=True), min=1e-12))

    # Normalize
    normalized_angles = angle_agglo / norm_denom

    # Make default frames
    default_frames = torch.tensor(rc.restype_rigid_group_default_frame, dtype=torch.float32,
                                  device=X.device, requires_grad=False)

    # Make group ids
    group_idx = torch.tensor(rc.restype_atom14_to_rigid_group, device=X.device,
                             requires_grad=False)

    # Make atom mask
    atom_mask = torch.tensor(rc.restype_atom14_mask, dtype=torch.float32,
                             device=X.device, requires_grad=False)

    # Make literature positions
    lit_positions = torch.tensor(rc.restype_atom14_rigid_group_positions, dtype=torch.float32,
                                 device=X.device, requires_grad=False)

    # Make all global frames
    all_frames_to_global = torsion_angles_to_frames(bb_to_global, normalized_angles, S, default_frames)

    # Predict coordinates
    pred_xyz = frames_and_literature_positions_to_atom14_pos(all_frames_to_global, S, default_frames, group_idx, 
                                                             atom_mask, lit_positions)
    
    # Replace backbone atoms with input coordinates
    pred_xyz[..., :4, :] = X[..., :4, :]

    return pred_xyz
