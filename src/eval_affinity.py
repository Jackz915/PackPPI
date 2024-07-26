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

from src.models.AffinityPrediction import AffinityPrediction
from src.datamodules.components.skempi_dataset import SkempiDataset
from src.utils.protein import from_pdb_file, to_pdb


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
    model = AffinityPrediction.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location=device,
        strict=False,
        encoder_cfg=hydra.utils.instantiate(cfg.model.encoder_cfg),
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        sample_cfg=hydra.utils.instantiate(cfg.model.sample_cfg)
    )
    return model.to(device).eval()

def evaluate_model(model: LightningModule, args: argparse.Namespace):
    print("----- Starting evaluation! -----")
    
    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq
        }

    muts = list(map(_parse_mut, args.mutstr.split(',')))

    protein = {
        'mutstr': args.mutstr,
        'num_muts': len(args.mutstr.split(',')),
        'mutations': muts,
        'pdb_path': args.input,
    }

    protein.update(vars(from_pdb_file(protein['pdb_path'], mse_to_met=True)))
    batch = SkempiDataset.prot_to_data(protein, cache_processed_data=False)

    for key in batch.keys():
            if not isinstance(batch[key], int):
                batch[key] = batch[key].unsqueeze(0)
                
    batch.num_proteins = 1
    batch.max_size = batch.num_nodes
    batch = batch.to(args.device)

    try:
        loss, pred = model.forward(batch)
    except RuntimeError as e:
        raise e

    pred = pred.cpu().item()
    print(f"----- The predicted binding affinity change (wildtype-mutant) is {pred:.4f} kcal/mol -----")
    print("----- Finishing evaluation! -----")

def main(args):
    cfg = load_configuration(config_path="../configs", config_name="eval_affinity.yaml")
    model = load_model(cfg, args.device)
    evaluate_model(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='The input pdb file path.', required=True)
    parser.add_argument('--mutstr', type=str, help='A string containing wild-type residue, chain ID, position, and mutant residue (e.g., "RA47A"). If more than one mutation, please separated by commas (e.g., "RA47A,EA48A").', required=True)
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    args = parser.parse_args()

    main(args)
