import os
import argparse
from pathlib import Path
import torch
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
import rootutils
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from typing import List, Optional, Tuple

import src.utils.residue_constants as rc
from src.models.TorsionalDiffusion import TDiffusionModule
from src.models.components import get_atom14_coords
from src.utils.protein import from_pdb_file, to_pdb
from src.utils.residue_constants import sidechain_atoms
from src.utils.protein_analysis import ProteinAnalysis


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def load_configuration(config_path: str, config_name: str) -> DictConfig:
    initialize(version_base="1.3", config_path=config_path)
    cfg = compose(config_name=config_name, return_hydra_config=True)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)
    return cfg

def load_model(cfg: DictConfig, device: str) -> LightningModule:
    assert cfg.get("ckpt_path") is not None and os.path.exists(cfg.ckpt_path), "Invalid checkpoint path!"
    
    print(f"----- Loading {cfg.ckpt_path} checkpoint! -----")
    model = TDiffusionModule.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location=device,
        strict=False,
        encoder_cfg=hydra.utils.instantiate(cfg.model.encoder_cfg),
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        sample_cfg=hydra.utils.instantiate(cfg.model.sample_cfg)
    )
    return model.to(device).eval()

def contains_sidechains(pdb_file: str) -> bool:
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name in sidechain_atoms:
                    return True
    return False

def evaluate_model(model: LightningModule, args: argparse.Namespace):
    print("----- Starting evaluation! -----")
    protein_analysis = ProteinAnalysis(args.molprobity_clash_loc, args.outdir, args.device)

    protein = vars(from_pdb_file(Path(args.input), mse_to_met=True))

    batch = protein_analysis.get_prot(args.input)
    batch = batch.to(args.device)

    try:
        SC_D_sample = model.sampling(batch, use_proximal=args.use_proximal)
    except RuntimeError as e:
        raise e

    predict_xyz = get_atom14_coords(batch.X, batch.residue_type, batch.BB_D, SC_D_sample)
    residx_atom37_to_atom14, atom37_atom_exists, _, _ = make_atom14_masks(batch.residue_type)
    predict_xyz = atom14_to_atom37(predict_xyz, residx_atom37_to_atom14, atom37_atom_exists)

    atom_positions = predict_xyz.cpu().squeeze().numpy()
    protein['atom_positions'] = atom_positions
    protein['atom_mask'] = (np.sum(atom_positions, axis=-1) != 0.0).astype(np.int32).squeeze()
    temp_protein = to_pdb(protein)

    with open(protein_analysis.tmp_pdb, 'w') as temp_file:
        temp_file.writelines(temp_protein)

    if contains_sidechains(args.input):
        metric = protein_analysis.get_metric(true_pdb=args.input, pred_pdb=protein_analysis.tmp_pdb)
        print(f"----- Metric: ----- {metric}")
    else:
        print("----- No side chain atoms found in the input PDB. Skipping metric calculation. -----")

    print("----- Finishing evaluation! -----")

def main(args):
    cfg = load_configuration(config_path="../configs", config_name="eval_diffusion.yaml")
    model = load_model(cfg, args.device)
    evaluate_model(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='The input pdb file path.', required=True)
    parser.add_argument('--outdir', type=str, help='Directory to store outputs.', required=True)
    parser.add_argument('--molprobity_clash_loc', type=str, help='Path to /build/bin/molprobity.clashscore.', required=True)
    parser.add_argument('--use_proximal', action="store_true", help='Use proximal optimize.')
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    args = parser.parse_args()

    main(args)
