import os
import numpy as np
import argparse
from pathlib import Path
import pyrootutils
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from typing import List, Optional, Tuple

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.optimize import proximal_optimizer
from src.models.components import get_atom14_coords
from src.utils.protein import from_pdb_file, to_pdb
from src.utils.residue_constants import sidechain_atoms
from src.utils.protein_analysis import ProteinAnalysis


def contains_sidechains(pdb_file: str) -> bool:
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name in sidechain_atoms:
                    return True
    return False

def main(args: argparse.Namespace):
    assert contains_sidechains(args.input),  "----- No side chain atoms found in the input PDB -----"
    
    print("----- Starting optimize! -----")
    protein_analysis = ProteinAnalysis(args.molprobity_clash_loc, args.outdir)

    clashscore = protein_analysis.get_clashscore(args.input)
    print(f"----- The input structure clashscore is {clashscore} -----")

    protein = {}
    protein.update(vars(from_pdb_file(Path(args.input), mse_to_met=True)))

    batch = protein_analysis.get_prot(args.input)

    try:
        SC_D_resample_list, loss_list = proximal_optimizer(batch, batch.SC_D,
                                                           args.violation_tolerance_factor,
                                                           args.clash_overlap_tolerance,
                                                           args.lamda,
                                                           args.num_steps)
            
    except RuntimeError as e:
        raise e

    if loss_list[-1] < loss_list[0]:
        SC_D_resample = SC_D_resample_list[-1]
    else:
        SC_D_resample = batch.SC_D
              
    predict_xyz = get_atom14_coords(batch.X, batch.residue_type, batch.BB_D, SC_D_resample)

    protein['atom_positions'] = predict_xyz.cpu().squeeze().numpy()
    temp_protein = to_pdb(protein)

    with open(protein_analysis.tmp_pdb, 'w') as temp_file:
        temp_file.writelines(temp_protein)

    clashscore = protein_analysis.get_clashscore(protein_analysis.tmp_pdb)
    print(f"----- The optimized structure clashscore is {clashscore} -----")

    print("----- Finishing optimize! -----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='The input pdb file path.', required=True)
    parser.add_argument('--outdir', type=str, help='Directory to store outputs.', required=True)
    parser.add_argument('--molprobity_clash_loc', type=str, help='Path to /build/bin/molprobity.clashscore.', required=True)
    parser.add_argument('--violation_tolerance_factor', type=float, help='The violation tolerance factor.', default=12)
    parser.add_argument('--clash_overlap_tolerance', type=float, help='Acceptable deviation between atoms.', default=0.5)
    parser.add_argument('--lamda', type=float, help='The influence of the proximal term on the gradient.', default=1)
    parser.add_argument('--num_steps', type=int, help='Number of optimize steps.', default=50)
    args = parser.parse_args()

    main(args)
