from typing import Any, List, Optional, Union
from pytorch_lightning import LightningModule
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from src.models.components.encoder import ProteinEncoder
from src.models.components.mpnn import MpnnNet
from src.utils.pylogger import get_pylogger
from src.models.TorsionalDiffusion import TDiffusionModule

log = get_pylogger(__name__)


class AffinityPrediction(LightningModule):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            encoder_cfg: DictConfig,
            model_cfg: DictConfig,
            sample_cfg: DictConfig,
            **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.valid_modes = ["network", "linear", "esm"]
        if self.hparams.mode not in self.valid_modes:
            raise ValueError(f"Invalid mode '{self.hparams.mode}'. Valid modes are: {self.valid_modes}.")

        if self.hparams.mode in ["network", "linear"]:
            self.pret = TDiffusionModule.load_from_checkpoint(checkpoint_path=self.hparams.ckpt_path,
                                                              map_location=self.device,
                                                              strict=False,
                                                              encoder_cfg=hydra.utils.instantiate(self.hparams.encoder_cfg),
                                                              model_cfg=hydra.utils.instantiate(self.hparams.model_cfg),
                                                              sample_cfg=hydra.utils.instantiate(self.hparams.sample_cfg))
            self.pret.freeze()
            self.pret_network = self.pret.network

        if self.hparams.mode == 'network':
            self.mutation_encoder = ProteinEncoder(node_in=self.hparams.encoder_cfg.node_in,
                                                   edge_in=self.hparams.encoder_cfg.edge_in,
                                                   node_features=self.hparams.encoder_cfg.node_features,
                                                   edge_features=self.hparams.encoder_cfg.edge_features,
                                                   time_embedding_type=self.hparams.encoder_cfg.time_embedding_type,
                                                   time_embedding_dim=0,
                                                   num_positional_embeddings=self.hparams.encoder_cfg.num_positional_embeddings,
                                                   num_rbf=self.hparams.encoder_cfg.num_rbf,
                                                   top_k=self.hparams.encoder_cfg.top_k,
                                                   af2_relpos=self.hparams.encoder_cfg.af2_relpos)

            self.mutation_mpnn = MpnnNet(
                node_features=self.hparams.encoder_cfg.node_features,
                edge_features=self.hparams.encoder_cfg.edge_features,
                hidden_dim=self.hparams.model_cfg.hidden_dim,
                num_mpnn_layers=self.hparams.model_cfg.num_mpnn_layers,
                n_points=self.hparams.model_cfg.n_points,
                dropout=self.hparams.model_cfg.dropout,
                act=self.hparams.model_cfg.act,
                position_scale=self.hparams.model_cfg.position_scale,
                use_ipmp=self.hparams.model_cfg.use_ipmp,
                k_neighbors=self.hparams.model_cfg.k_neighbors)

            self.mut_bias = nn.Embedding(
                num_embeddings = 2,
                embedding_dim = self.hparams.model_cfg.hidden_dim,
                padding_idx = 0,
            )
            
            self.mutation_fusion = nn.Sequential(
                nn.Linear(2*self.hparams.model_cfg.hidden_dim, self.hparams.model_cfg.hidden_dim), nn.ReLU(),
                nn.Linear(self.hparams.model_cfg.hidden_dim, self.hparams.model_cfg.hidden_dim)
            )
                
        self.ddg_predictor = nn.Sequential(
            nn.Linear(self.hparams.model_cfg.hidden_dim, self.hparams.model_cfg.hidden_dim), nn.ReLU(),
            nn.Linear(self.hparams.model_cfg.hidden_dim, self.hparams.model_cfg.hidden_dim), nn.ReLU(),
            nn.Linear(self.hparams.model_cfg.hidden_dim, 1)
        )

        # Initialization
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        # metrics
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.train_phase, self.val_phase = "train", "val"

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def get_local_subgraph(self, X, mut_mask, radius=10):
        batch_size, num_residues, num_atoms = X.shape
    
        # Flatten the atomic coordinates for distance calculation
        X_flat = X.view(batch_size, num_residues, -1)
    
        # Compute pairwise Euclidean distances
        dist_matrix = torch.cdist(X_flat, X_flat) 
    
        # Generate local mask based on distance threshold
        local_mask = dist_matrix < radius  
        
        # Convert mut_mask to the same shape as local_mask and to torch.uint8
        mut_mask_expanded = mut_mask.unsqueeze(1).expand(batch_size, num_residues, num_residues).to(torch.uint8)
    
        # Combine with the mutation mask using bitwise AND
        combined_mask = local_mask & mut_mask_expanded
    
        # Reduce to a single mask per residue by taking the maximum along the residue dimension
        local_mask = combined_mask.any(dim=2).to(torch.float32)  
    
        return local_mask
        
    @torch.no_grad()
    def get_pret_feature(self, batch):
        t = torch.tensor([0.],
                         requires_grad=False).repeat_interleave(batch.num_proteins * batch.max_size).to(self.device)
        _, h_V_pret = self.pret_network(batch, batch.SC_D, t)
        return h_V_pret

    def encode(self, batch):    
        X_ca = batch.X[:, :, 1, :]
        mut_mask = batch.mut_mask
        local_mask = self.get_local_subgraph(X_ca, mut_mask)
        
        h_V_mutation, h_E, E_idx, X = self.mutation_encoder(batch.X,
                                                            batch.residue_type,
                                                            batch.BB_D_sincos,
                                                            batch.SC_D_sincos,
                                                            batch.chain_indices,
                                                            local_mask,
                                                            batch.residue_index)

        h_V_pret = self.get_pret_feature(batch)
        h_V = self.mutation_fusion(torch.cat([h_V_pret, h_V_mutation], dim=-1))

        bias = self.mut_bias(batch['mut_mask'])
        h_V = h_V + bias
        
        h_V = self.mutation_mpnn(h_V, h_E, E_idx, X, batch.residue_type, local_mask)
        return h_V

    def forward(self, batch: Any):
        if self.hparams.mode == 'esm':
            h_wt = batch['esm_representations']
            h_mt = batch['esm_representations_mut']

        else:
            batch_mt = batch.clone()
            for key in ['atom_mask', 'residue_type', 'SC_D', 'SC_D_sincos', 'SC_D_mask',
                        'chi_1pi_periodic_mask', 'chi_2pi_periodic_mask']:
                batch_mt[key] = batch_mt[key + '_mut']
    
            if self.hparams.mode == 'network':
                h_wt = self.encode(batch)
                h_mt = self.encode(batch_mt)
            else:
                h_wt = self.get_pret_feature(batch)
                h_mt = self.get_pret_feature(batch_mt)

        ddg_pred = self.ddg_predictor((h_mt - h_wt).max(dim=1)[0]) # mean(dim=1)
        ddg_pred_inv = self.ddg_predictor((h_mt - h_wt).max(dim=1)[0]) # mean(dim=1)
        
        labels = batch['ddg']
        loss = (self.criterion(ddg_pred.squeeze(-1), labels) + self.criterion(ddg_pred_inv.squeeze(-1), -labels)) / 2
        return loss, ddg_pred

    def step(self, batch: Any):
        loss, ddg_pred = self.forward(batch)

        return loss, ddg_pred

    def on_train_start(self):
        self.val_loss.reset()

    def training_step(self, batch: Any, batch_idx: int):
        try:
            loss, ddg_pred = self.step(batch)

        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping training batch with index {batch_idx} due to OOM error...")
            return

        # skip backpropagation if loss was invalid
        if loss.isnan().any() or loss.isinf().any():
            log.info(f"Loss for batch with index {batch_idx} is invalid. Skipping...")
            return

        # update metric(s)
        self.train_loss(loss.detach())

        return {"loss": loss, "label": ddg_pred}

    def on_train_epoch_end(self):
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=True)

    def on_validation_start(self):
        return

    def validation_step(self, batch: Any, batch_idx: int):
        try:
            loss, ddg_pred = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # update metric(s)
        self.val_loss(loss.detach())

        return {"loss": loss, "label": ddg_pred}

    def on_validation_epoch_end(self):
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # def configure_gradient_clipping(
    #     self,
    #     optimizer: torch.optim.Optimizer,
    #     gradient_clip_val: Optional[Union[int, float]] = None,
    #     gradient_clip_algorithm: Optional[str] = None,
    #     verbose: bool = False
    #     ):

    #     self.clip_gradients(
    #         optimizer,
    #         gradient_clip_val=1,
    #         gradient_clip_algorithm="norm"
    #     )
