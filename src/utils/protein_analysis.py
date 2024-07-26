import os
import subprocess
import torch
import numpy as np
from pathlib import Path
from src.datamodules.components.complex_dataset import ComplexDataset
from src.datamodules.components.helper import *
from src.models.components import get_atom14_coords
from src.utils.protein import from_pdb_file, to_pdb


class ProteinAnalysis:
    def __init__(self, molprobity_clash_loc, tmp_dir, device='cpu', 
                 scwrl_loc=None, faspr_loc=None, rosetta_loc=None):
        self.molprobity_clash_loc = molprobity_clash_loc
        self.scwrl_loc = scwrl_loc
        self.faspr_loc = faspr_loc
        self.rosetta_loc = rosetta_loc
        self.device = device

        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.tmp_log = os.path.join(tmp_dir, 'molprobity_clash.log')
        self.tmp_pdb = os.path.join(tmp_dir, 'structure.pdb')

    def get_clashscore(self, pdb):
        clash_line = f'{self.molprobity_clash_loc} model={pdb} keep_hydrogens=True > {self.tmp_log}'
        subprocess.run(clash_line, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        command = f'grep "clashscore" {self.tmp_log} | awk -F "= " \'{{print $2}}\''
        clashscore_bytes = subprocess.check_output(command, shell=True)
        clashscore_str = clashscore_bytes.decode().strip()

        return float(clashscore_str) if clashscore_str.replace('.', '').isdigit() else None

    def get_metric(self, true_pdb, pred_pdb):
        try:
            true_data = self.get_prot(true_pdb, get_interface=True)
            pred_data = self.get_prot(pred_pdb)
        except Exception as e:
            print(f"Error: Failed to load or parse PDB files. Details: {str(e)}")
            return None
    
        if true_data.X.shape[0] != pred_data.X.shape[0]:
            print("Error: Mismatch in the number of residues between true and predicted structures.")
            return None
    
        clashscore = self.get_clashscore(pred_pdb)
        if clashscore is None:
            print("Error: Failed to calculate clash score for the predicted structure.")
            return None
    
        interface_mask = true_data.interface_mask
        chis_true, chis_pred, chi_mask, chi_1pi_periodic_mask = (
            true_data.SC_D, pred_data.SC_D, true_data.SC_D_mask, true_data.chi_1pi_periodic_mask
        )
        metric = {}
        total_acc = 0
        interface_acc = 0

        for i in range(4):
            chis_true_ = chis_true[..., i]
            chis_pred_ = chis_pred[..., i]
            chi_num = 1 if chi_mask[..., i].sum() == 0 else chi_mask[..., i].sum()
            interface_num = 1 if (chi_mask[..., i] * interface_mask).sum() == 0 else (chi_mask[..., i] * interface_mask).sum()
    
            chi_1pi_periodic_mask_ = chi_1pi_periodic_mask[..., i]
    
            chi_diff = (chis_pred_ - chis_true_).abs()
    
            condition = torch.logical_and(chi_diff * 180 / np.pi < 20, chi_diff > 0)
            chi_acc = torch.where(condition, 1., 0.)
            chi_ae = torch.minimum(chi_diff, 2 * np.pi - chi_diff)
            chi_ae_periodic = torch.minimum(chi_ae, np.pi - chi_ae)
            chi_ae[chi_1pi_periodic_mask_] = chi_ae_periodic[chi_1pi_periodic_mask_]
            chi_ae_rad = chi_ae
            chi_ae_deg = chi_ae * 180 / np.pi
    
            metric[f"chi_{i}_ae_rad"] = chi_ae_rad.sum() / chi_num
            metric[f"chi_{i}_ae_deg"] = chi_ae_deg.sum() / chi_num
            metric[f"chi_{i}_acc"] = chi_acc.sum() / chi_num
            total_acc += chi_acc.sum() / chi_num
            interface_acc += (chi_acc * interface_mask).sum() / interface_num
    
        predict_coords = get_atom14_coords(true_data.X, true_data.residue_type, true_data.BB_D, pred_data.SC_D)
        predict_coords[..., :5, :] = true_data.X[..., :5, :]
        metric["atom_rmsd"] = self.compute_rmsd(true_data.X, predict_coords, true_data.atom_mask, true_data.residue_mask)
        metric["total_acc"] = total_acc / 4
        metric["interface_acc"] = interface_acc / 4
        metric["clashscore"] = clashscore

        return metric

    def compute_rmsd(self, true_coords, pred_coords, atom_mask, residue_mask, eps=1e-6):
        per_atom_sq_err = torch.sum((true_coords - pred_coords) ** 2, dim=-1) * atom_mask * residue_mask[..., None]
        per_res_sq_err = torch.sum(per_atom_sq_err, dim=-1)
        per_res_atom_count = torch.sum(atom_mask * residue_mask[..., None] + eps, dim=-1)

        total_sq_err = torch.sum(per_res_sq_err)
        total_atom_count = torch.sum(per_res_atom_count)
        rmsd = total_sq_err / total_atom_count
        return rmsd

    def get_prot(self, pdb, get_interface=True):
        protein = {}
        protein.update(vars(from_pdb_file(Path(pdb), mse_to_met=True)))
        data = ComplexDataset.prot_to_data(protein, cache_processed_data=False)
        if get_interface:
            interface_mask = get_interface_mask(protein, pdb=pdb)

            if interface_mask is not None:
                data['interface_mask'] = interface_mask * data.residue_mask
            else:
                data['interface_mask'] = torch.zeros_like(data.residue_mask)

        for key in data.keys():
            if not isinstance(data[key], int):
                data[key] = data[key].unsqueeze(0)
                
        data.num_proteins = 1
        data.max_size = data.num_nodes
        
        return data

    def run_tool(self, in_pdb, tool_name):
        if tool_name == 'scwrl' and self.scwrl_loc:
            scwrl_line = f'{self.scwrl_loc} -i {in_pdb} -o {self.tmp_pdb}'
            subprocess.run(scwrl_line, shell=True, stdout=subprocess.DEVNULL)

        elif tool_name == 'faspr' and self.faspr_loc:            
            faspr_line = f'{self.faspr_loc} -i {in_pdb} -o {self.tmp_pdb}'
            subprocess.run(faspr_line, shell=True, stdout=subprocess.DEVNULL)

        elif tool_name == 'rosetta' and self.rosetta_loc:            
            self.tmp_pdb = f'{self.rosetta_loc}' + os.path.basename(in_pdb)  
            
        else:
            raise ValueError("Invalid tool name")

        metric = self.get_metric(in_pdb, self.tmp_pdb)
        return metric
