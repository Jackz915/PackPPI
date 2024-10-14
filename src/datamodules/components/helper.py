import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from typing import *
from torch.nn import functional as F
import src.utils.residue_constants as rc
from src.utils.interface import get_interface_residues, extract_interface, parse_interface_file


def _normalize(tensor: torch.Tensor, dim: int = -1):
    """
    From https://github.com/drorlab/gvp-pytorch
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def calc_dihedrals(atom_positions: torch.Tensor, eps=1e-8) -> torch.Tensor:

    # Unit vectors
    uvecs = _normalize(atom_positions[..., 1:, :] - atom_positions[..., :-1, :])
    uvec_2 = uvecs[..., :-2, :]
    uvec_1 = uvecs[..., 1:-1, :]
    uvec_0 = uvecs[..., 2:, :]

    nvec_2 = _normalize(torch.cross(uvec_2, uvec_1, dim=-1))
    nvec_1 = _normalize(torch.cross(uvec_1, uvec_0, dim=-1))

    # Angle between normals
    cos_dihedral = torch.sum(nvec_2 * nvec_1, dim=-1)
    cos_dihedral = torch.clamp(cos_dihedral, -1 + eps, 1 - eps)
    dihedral = torch.sign(torch.sum(uvec_2 * nvec_1, dim=-1)) * torch.acos(cos_dihedral)

    return dihedral


def calc_bb_dihedrals(atom_positions: torch.Tensor,
                      residue_index: Optional[torch.Tensor] = None,
                      use_pre_omega: bool = True,
                      return_mask: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    # Get backbone coordinates (and reshape). First 3 coordinates are N, CA, C
    bb_atom_positions = atom_positions[:, :3].reshape((3 * atom_positions.shape[0], 3))

    # Get backbone dihedrals
    bb_dihedrals = calc_dihedrals(bb_atom_positions)

    bb_dihedrals = F.pad(bb_dihedrals, [1, 2], value=torch.nan)  # Add empty phi[0], psi[-1], and omega[-1]
    bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))

    # Get mask based on residue_index
    bb_dihedrals_mask = torch.ones_like(bb_dihedrals)
    if residue_index is not None:
        pre_mask = torch.cat(
            (torch.tensor([0.0]), (residue_index[1:] - 1 == residue_index[:-1]).to(torch.float32)), dim=-1)
        post_mask = torch.cat(
            ((residue_index[:-1] + 1 == residue_index[1:]).to(torch.float32), torch.tensor([0.0])), dim=-1)
        bb_dihedrals_mask = torch.stack((pre_mask, post_mask, post_mask), dim=-1)

    if use_pre_omega:
        # Move omegas such that they're "pre-omegas" and reorder dihedrals
        bb_dihedrals[:, 2] = torch.cat((torch.tensor([torch.nan]), bb_dihedrals[:-1, 2]), dim=-1)
        bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
        bb_dihedrals_mask[:, 1] = bb_dihedrals_mask[:, 0]

    # Update dihedral_mask
    bb_dihedrals_mask = bb_dihedrals_mask * torch.isfinite(bb_dihedrals).to(torch.float32)

    if return_mask:
        return bb_dihedrals, bb_dihedrals_mask
    else:
        return bb_dihedrals


def calc_sc_dihedrals(atom_positions: torch.Tensor,
                      aatype: torch.Tensor,
                      return_mask: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    # Make sure atom_positions and aatype are same class
    chi_atom_indices = torch.from_numpy(np.array(rc.chi_atom_indices_atom14, dtype=np.int32)).to(aatype.device)[
        aatype]
    chi_mask = torch.from_numpy(np.array(rc.chi_mask_atom14, dtype=np.float32)).to(aatype.device)[aatype]

    # Get coordinates for chi atoms
    chi_atom_positions = torch.gather(atom_positions, -2,
                                      chi_atom_indices[..., None].expand(*chi_atom_indices.shape, 3).long())

    sc_dihedrals = calc_dihedrals(chi_atom_positions)

    # Chi angles that are missing an atom will be NaN, so turn all those to 0.
    sc_dihedrals = torch.nan_to_num(sc_dihedrals)

    sc_dihedrals = sc_dihedrals * chi_mask
    sc_dihedrals_mask = (sc_dihedrals != 0.).to(torch.float32)

    if return_mask:
        return sc_dihedrals, sc_dihedrals_mask
    else:
        return sc_dihedrals


def get_interface_mask(protein: Dict,
                       pdb,
                       radius=10.0) -> torch.Tensor:

    if len(np.unique(protein["chain_id"])) == 1:
        return None
        
    # for chain in protein["chain_id"]:
    #     _ = extract_interface(protein, chain, outdir=cache_path)
    # inter_residues = parse_interface_file(cache_path)

    inter_residues = get_interface_residues(pdb)
    inter_chain = set(protein["chain_id"]) & set([i for i in inter_residues.keys()])
    inter_residues_mask = []
    for id in np.unique(protein["chain_id"]):
        sub_residue = protein["residue_index"][protein["chain_id"] == id]
        if id not in inter_chain:
            chain_residue_mask = np.full(len(sub_residue), False, dtype=bool)
            inter_residues_mask.append(chain_residue_mask)
        else:
            chain_residue_mask = np.isin(sub_residue, inter_residues[id])
            inter_residues_mask.append(chain_residue_mask)
    inter_residues_mask = np.concatenate(inter_residues_mask)

    return torch.from_numpy(inter_residues_mask).to(torch.float32)


def get_esm_feature(sequence,
                    chain,
                    residues_mask: Optional[Any] = None,
                    padding_length: int = 20,
                    esm_model: Optional[nn.Module] = None,
                    esm_batch_converter: Optional[Any] = None
                    ) -> torch.Tensor:

    protein_sequence = []
    for i in torch.unique(chain):
        sub_residue = sequence[chain == i].tolist()
        if residues_mask is not None:
            sub_residue_mask = residues_mask[chain == i].tolist()
            for j in range(0, len(sub_residue)):
                protein_sequence.append('<mask>' if sub_residue_mask[j] else sub_residue[j])
        else:
            for j in range(0, len(sub_residue)):
                protein_sequence.append(sub_residue[j])

        if i == max(torch.unique(chain)):
            continue
        else:
            protein_sequence.append('<pad>' * padding_length)
    protein_sequence = [('', ''.join(protein_sequence))]

    # add ESM sequence embeddings as scalar atom features that are shared between atoms of the same residue
    batch_tokens = esm_batch_converter(protein_sequence)[2]
    batch_lens = sequence.shape[0]

    with torch.inference_mode():
        results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers])
    token_representations = results["representations"][esm_model.num_layers].cpu()
    protein_representations = []
    for i, (_, protein_sequence) in enumerate(protein_sequence):
        representations = token_representations[i, 1: batch_lens + 1]
        protein_representations.append(representations)
    protein_representations = torch.cat(protein_representations, dim=0)

    return protein_representations


