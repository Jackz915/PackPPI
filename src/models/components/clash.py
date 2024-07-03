import src.utils.residue_constants as rc
import torch
import torch.nn.functional as F
from src.models.components import get_atom14_coords, get_sc_atom14_mask


def within_residue_violations(
    atom14_pred_positions,
    atom14_atom_exists,
    atom14_dists_lower_bound,
    atom14_dists_upper_bound,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
):
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Backbone-backbone clashes are ignored. CB is included in the backbone.
    bb_bb_mask = torch.zeros_like(dists_masks)
    bb_bb_mask[..., :5, :5] = 1.0
    dists_masks = dists_masks * (1.0 - bb_bb_mask)

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    per_atom_num_clash = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
        "per_atom_num_clash": per_atom_num_clash
    }


def between_residue_clash_loss(
        atom14_pred_positions,
        atom14_atom_exists,
        atom14_atom_radius,
        residue_index,
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5,
        eps=1e-10,
):
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                    atom14_pred_positions[..., :, None, :, None, :]
                    - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
            atom14_atom_exists[..., :, None, :, None]
            * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Backbone-backbone clashes are ignored. CB is included in the backbone.
    bb_bb_mask = torch.zeros_like(dists_mask)
    bb_bb_mask[..., :5, :5] = 1.0
    dists_mask = dists_mask * (1.0 - bb_bb_mask)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
            residue_index[..., :, None, None, None]
            < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None] + 1) == residue_index[..., None, :]

    neighbour_mask = neighbour_mask[..., None, None]

    c_n_bonds = (
            neighbour_mask
            * c_one_hot[..., None, None, :, None]
            * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
            cys_sg_one_hot[..., None, None, :, None]
            * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
            atom14_atom_radius[..., :, None, :, None]
            + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
            dists < (dists_lower_bound - overlap_tolerance_hard)
    )
    per_atom_num_clash = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(clash_mask, dim=(-3, -1))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
        "per_atom_num_clash": per_atom_num_clash,  # shape (N, 14),
    }


def find_sc_violations(atom14_pred_positions,
                       atom14_atom_exists,
                       residue_type,
                       residue_index,
                       violation_tolerance_factor: float,
                       clash_overlap_tolerance: float):
    restype_atom14_to_atom37 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.long,
        device=residue_type.device
    )

    residx_atom14_to_atom37 = restype_atom14_to_atom37[residue_type]

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (*, N, 14).
    atomtype_radius = [
        rc.van_der_waals_radius[name[0]]
        for name in rc.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        atom14_atom_exists
        * atomtype_radius[residx_atom14_to_atom37]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[residue_type]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[residue_type]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    return {
        "between_residues": {
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
        }
    }


def compute_residue_clash(batch,
                          SC_D,
                          violation_tolerance_factor=12.,
                          clash_overlap_tolerance=0.5,
                          eps=1e-10):

    # mask the backbone atom and get per_residue_atoms
    atom_mask_modified = batch.atom_mask.clone()  # [B, N, 14]
    atom_mask_modified[..., :5] = 0
    per_residue_atoms = torch.sum(atom_mask_modified, dim=(-1))  # [B, N]

    # get atom14_coords
    atom14_coords = get_atom14_coords(batch.X, batch.residue_type, batch.BB_D, SC_D)
    atom14_coords[..., :5, :] = batch.X[..., :5, :]

    clash_info = find_sc_violations(atom14_pred_positions=atom14_coords,
                                    atom14_atom_exists=batch.atom_mask,
                                    residue_type=batch.residue_type,
                                    residue_index=batch.residue_index,
                                    violation_tolerance_factor=violation_tolerance_factor,
                                    clash_overlap_tolerance=clash_overlap_tolerance)

    per_atom_clash = (clash_info["between_residues"]["clashes_per_atom_loss_sum"] +
                      clash_info["within_residues"]["per_atom_loss_sum"])  # [B, N, 14]

    # mask backbone atoms loss
    per_atom_clash[..., :5] = 0

    # scale by per_residue_atoms
    per_residue_clash = torch.sum(per_atom_clash, dim=(-1)) / (eps + per_residue_atoms) # [B, N]
    
    return per_residue_clash

    # sc_clash = []
    # for chi_id in range(self.NUM_CHI_ANGLES):
    #     sc_atom14_mask = get_sc_atom14_mask(batch.residue_type, chi_id)
    #     residue_atoms = torch.sum(batch.atom_mask * sc_atom14_mask, dim=-1)
    
    #     per_atom_clash = (clash_info["between_residues"]["clashes_per_atom_loss_sum"] * sc_atom14_mask +
    #                       clash_info["within_residues"]["per_atom_loss_sum"] * sc_atom14_mask)  # [B, N, 14]
    #     per_atom_clash[..., :5] = 0
    #     per_residue_clash = torch.sum(per_atom_clash, dim=(-1))
    #     sc_clash.append((per_residue_clash / (self.eps + residue_atoms)).unsqueeze(-1))
    
    # sc_clash = torch.cat(sc_clash, dim=-1) * batch.SC_D_mask
    # return sc_clash
