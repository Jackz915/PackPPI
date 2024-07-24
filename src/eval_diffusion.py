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

from src.models.TorsionalDiffusion import TDiffusionModule
from src.models.components import get_atom14_coords
from src.utils.protein import from_pdb_file, to_pdb
from src.utils.protein_analysis import ProteinAnalysis


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

log = utils.get_pylogger(__name__)


def main(args):
    initialize(version_base="1.3", config_path="../configs")
    # initialize(config_path="configs")
    cfg = compose(config_name="eval_diffusion.yaml", return_hydra_config=True)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)

    assert cfg.get("ckpt_path") is not None and os.path.exists(cfg.ckpt_path), "Invalid checkpoint path!"
    
    print("----- Loading checkpoint! -----")
    model = TDiffusionModule.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location=args.device,
        strict=False,
        encoder_cfg=hydra.utils.instantiate(cfg.model.encoder_cfg),
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        sample_cfg=hydra.utils.instantiate(cfg.model.sample_cfg)
    )
    model = model.to(args.device)
    model.eval()
    
    print("----- Starting evaluation! -----")
    protein_analysis = ProteinAnalysis(args.molprobity_clash_loc, 
                                       args.outdir, 
                                       args.device)

    protein = {}
    protein.update(vars(from_pdb_file(Path(args.input), mse_to_met=True)))

    batch = protein_analysis.get_prot(args.input)
    batch = batch.to(args.device)

    try:
        if args.use_proximal:
            SC_D_sample, _ = model.sampling(batch, use_proximal=args.use_proximal, return_list=False)
        else:
            SC_D_sample = model.sampling(batch, use_proximal=args.use_proximal, return_list=False)
            
    except RuntimeError as e:
        raise (e)

    predict_xyz = get_atom14_coords(batch.X, batch.residue_type, batch.BB_D, SC_D_sample)
    predict_xyz[..., :5, :] = batch.X[..., :5, :]
    
    protein['atom_positions'] = predict_xyz.cpu().squeeze().numpy()
    temp_protein = to_pdb(protein)
     
    with open(protein_analysis.tmp_pdb, 'w') as temp_file:
        for line in temp_protein:
            temp_file.writelines(line)

    metric = protein_analysis.get_metric(true_pdb=args.input, pred_pdb=protein_analysis.tmp_pdb)
    
    print("----- Metric: -----", metric)
    print("----- Finishing evaluation! -----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='The input pdb file path.', required=True)
    parser.add_argument('--outdir', type=str, help='Directory to store outputs.', required=True)
    parser.add_argument('--molprobity_clash_loc', type=str, help='Path to /build/bin/molprobity.clashscore.', required=True)
    parser.add_argument('--use_proximal', action="store_true", help='Use proximal optimize.')
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    args = parser.parse_args()

    main(args)
    
