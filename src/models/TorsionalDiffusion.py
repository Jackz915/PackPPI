from typing import Any, Dict
from time import time
from omegaconf import DictConfig
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from src.datamodules.components.helper import *
from src.models.components import get_atom14_coords
from src.models.components.schedule import SO2VESchedule
from src.models.components.encoder import ProteinEncoder
from src.models.components.mpnn import MpnnNet
from src.models.components.layers import MLP
from src.models.components.clash import compute_residue_clash
from src.models.components.optimize import proximal_optimizer
from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class TDiffusionModule(LightningModule):
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

        self.NUM_CHI_ANGLES = 4
        self.eps = 1e-6

        self.save_hyperparameters(logger=False)

        # encoder
        self.encoder = ProteinEncoder(node_in=self.hparams.encoder_cfg.node_in,
                                      edge_in=self.hparams.encoder_cfg.edge_in,
                                      node_features=self.hparams.encoder_cfg.node_features,
                                      edge_features=self.hparams.encoder_cfg.edge_features,
                                      time_embedding_type=self.hparams.encoder_cfg.time_embedding_type,
                                      time_embedding_dim=self.hparams.encoder_cfg.time_embedding_dim,
                                      num_positional_embeddings=self.hparams.encoder_cfg.num_positional_embeddings,
                                      num_rbf=self.hparams.encoder_cfg.num_rbf,
                                      top_k=self.hparams.encoder_cfg.top_k,
                                      af2_relpos=self.hparams.encoder_cfg.af2_relpos)

        self.mpnn = MpnnNet(
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

        self.decoder_score = nn.ModuleList([MLP(self.hparams.model_cfg.hidden_dim,
                                                self.hparams.model_cfg.hidden_dim // 2,
                                                self.hparams.model_cfg.hidden_dim // 4, 2),
                                            nn.ReLU(),
                                            MLP(self.hparams.model_cfg.hidden_dim // 4,
                                                self.hparams.model_cfg.hidden_dim // 8,
                                                4, 2)])

        self.schedule_1pi_periodic = SO2VESchedule(pi_periodic=True,
                                                   annealed_temp=self.hparams.sample_cfg.annealed_temp,
                                                   mode=self.hparams.sample_cfg.mode)
        self.schedule_2pi_periodic = SO2VESchedule(pi_periodic=False,
                                                   annealed_temp=self.hparams.sample_cfg.annealed_temp,
                                                   mode=self.hparams.sample_cfg.mode)

        self.schedule = self.schedule_1pi_periodic.reverse_t_schedule.to(self.device)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # metrics
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

    def network(self, batch, SC_D_noised, t):
        SC_D_sincos_noised = torch.stack((torch.sin(SC_D_noised), torch.cos(SC_D_noised)), dim=-1)
        SC_D_sincos_noised = SC_D_sincos_noised * batch.SC_D_mask[..., None]

        h_V, h_E, E_idx, X = self.encoder(batch.X,
                                          batch.residue_type,
                                          batch.BB_D_sincos,
                                          SC_D_sincos_noised,
                                          batch.chain_indices,
                                          batch.residue_mask,
                                          batch.residue_index,
                                          t)

        h_V = self.mpnn(h_V, h_E, E_idx, X, batch.residue_type, batch.residue_mask)

        # get predict score
        pred_score = h_V.clone()
        for layer in self.decoder_score:
            pred_score = layer(pred_score)
        return pred_score, h_V

    @torch.no_grad()
    def add_sc_noise(self, batch, t):
        SC_D = batch.SC_D.reshape(-1, 4)
        chi_1pi_periodic_mask = batch.chi_1pi_periodic_mask.reshape(-1, 4)
        chi_2pi_periodic_mask = batch.chi_2pi_periodic_mask.reshape(-1, 4)

        SC_D_noised, score_1pi = self.schedule_1pi_periodic.add_noise(SC_D, t, chi_1pi_periodic_mask)
        SC_D_noised, score_2pi = self.schedule_2pi_periodic.add_noise(SC_D_noised, t, chi_2pi_periodic_mask)

        # restrict SC_D_noised from -pi to pi
        SC_D_noised = (SC_D_noised + np.pi) % (2 * np.pi) - np.pi

        score = torch.where(chi_1pi_periodic_mask, score_1pi, score_2pi)
        return SC_D_noised.reshape(batch.num_proteins, -1, 4), score.reshape(batch.num_proteins, -1, 4)

    def forward(self, batch: Any):
        t = self.schedule_1pi_periodic.sample_train_t(shape=(batch.num_proteins,
                                                             )).repeat_interleave(batch.max_size).to(self.device)
        sigma = self.schedule_1pi_periodic.t_to_sigma(t)

        # add noise to side-chain
        SC_D_noised, target_score = self.add_sc_noise(batch, t)

        pred_score, _ = self.network(batch, SC_D_noised, t)

        # scale predict score
        torsion_sigma = sigma.repeat_interleave(4).unsqueeze(-1).reshape(-1, 4)

        score_norm_1pi = torch.tensor(self.schedule_1pi_periodic.score_norm(torsion_sigma),
                                      requires_grad=False,
                                      device=self.device)
        score_norm_2pi = torch.tensor(self.schedule_2pi_periodic.score_norm(torsion_sigma),
                                      requires_grad=False,
                                      device=self.device)

        score_norm = torch.where(batch.chi_1pi_periodic_mask.reshape(-1, 4), score_norm_1pi, score_norm_2pi)
        score_norm = score_norm.reshape(batch.num_proteins, -1, 4)
        pred_score = pred_score * torch.sqrt(score_norm) * batch.SC_D_mask

        # get loss
        chi_sum = batch.SC_D_mask.sum() if batch.SC_D_mask.sum() > 0 else 1
        score_loss = ((target_score - pred_score) ** 2 / (score_norm + self.eps)).sum() / chi_sum
        return score_loss

    def step(self, batch: Any):
        loss = self.forward(batch)
        return loss

    def on_train_start(self):
        self.val_loss.reset()

    def training_step(self, batch: Any, batch_idx: int):
        try:
            loss = self.step(batch)

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
        return {"loss": loss}

    def on_train_epoch_end(self):
        # log metric(s)
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=True)

    def on_validation_start(self):
        self.data_iter = iter(self.trainer.datamodule.val_dataloader())

    def validation_step(self, batch: Any, batch_idx: int):
        try:
            loss = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # update metric(s)
        self.val_loss(loss.detach())
        return {"loss": loss}

    @rank_zero_only
    def log_evaluation_metrics(self, metrics_dict: Dict[str, Any]):
        for m, value in metrics_dict.items():
            self.log(f"{self.val_phase}/{m}", value, sync_dist=False)

    @torch.inference_mode()
    def evaluate_sampling(self):
        ticker = time()

        sampling_results = self.sample_and_analyze()
        self.log_evaluation_metrics(sampling_results)

        log.info(f"validation_epoch_end(): Sampling evaluation took {time() - ticker:.2f} seconds")

    def on_validation_epoch_end(self):
        # log metric(s)
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=True)

        time_to_evalute_sampling = (
                self.hparams.sample_cfg.sample_during_training
                and ((self.current_epoch + 1) % self.hparams.sample_cfg.eval_epochs == 0)
        )

        if time_to_evalute_sampling:
            self.evaluate_sampling()

    def test_step(self, batch: Any, batch_idx: int):
        try:
            loss = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            torch.cuda.empty_cache()
            log.info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # update metric(s)
        self.test_loss(loss.detach())
        return {"loss": loss}

    def on_test_epoch_end(self):
        # log metric(s)
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=True)

    @torch.inference_mode()
    def sample_and_analyze(self):
        dummy_batch = next(self.data_iter).to(self.device)
        SC_D_sample = self.sampling(dummy_batch)
        metric = self.analyze_samples(dummy_batch, SC_D_sample=SC_D_sample)
        return metric

    def sampling(self, batch: Any, use_proximal: bool = False, return_list: bool = False):
        t = torch.tensor([1.]).repeat_interleave(batch.max_size * batch.num_proteins).to(self.device)
        SC_D_sample, _ = self.add_sc_noise(batch, t)

        # multi-round sampling
        for j in range(len(self.schedule) - 1):
            time = self.schedule[j]
            dt = self.schedule[j] - self.schedule[j + 1] if j + 1 < len(self.schedule) else 1

            time, dt = time.to(self.device), dt.to(self.device)
            t = time.repeat_interleave(batch.num_proteins * batch.max_size)

            sample_score, _ = self.network(batch, SC_D_sample, t)

            # step backward
            sample_score = sample_score.reshape(-1, 4)
            SC_D_sample = SC_D_sample.reshape(-1, 4)

            SC_D_sample = self.schedule_1pi_periodic.step(SC_D_sample, sample_score, time, dt,
                                                          batch.chi_1pi_periodic_mask.reshape(-1, 4))
            SC_D_sample = self.schedule_2pi_periodic.step(SC_D_sample, sample_score, time, dt,
                                                          batch.chi_2pi_periodic_mask.reshape(-1, 4))

            # restrict SC_D_pred from -pi to pi
            SC_D_sample = (SC_D_sample + np.pi) % (2 * np.pi) - np.pi
            SC_D_sample = SC_D_sample.reshape(batch.num_proteins, -1, 4)
            SC_D_sample = SC_D_sample * batch.SC_D_mask

        if not use_proximal:
            return SC_D_sample

        SC_D_resample = proximal_optimizer(batch, SC_D_sample,
                                           self.hparams.sample_cfg.violation_tolerance_factor,
                                           self.hparams.sample_cfg.clash_overlap_tolerance,
                                           self.hparams.sample_cfg.lamda,
                                           self.hparams.sample_cfg.num_steps)
        
        return SC_D_resample

    def compute_rmsd(self, true_coords, pred_coords, atom_mask, residue_mask):
        per_atom_sq_err = torch.sum((true_coords - pred_coords) ** 2, dim=-1) * atom_mask * residue_mask[..., None]
        per_res_sq_err = torch.sum(per_atom_sq_err, dim=-1)
        per_res_atom_count = torch.sum(atom_mask * residue_mask[..., None] + self.eps, dim=-1)

        total_sq_err = torch.sum(per_res_sq_err)
        total_atom_count = torch.sum(per_res_atom_count)
        rmsd = total_sq_err / total_atom_count
        # rmsd = torch.sqrt(total_sq_err / total_atom_count)
        return rmsd

    def analyze_samples(self, batch, SC_D_sample=None):
        SC_D_true = batch['SC_D'].clone()
        SC_D_sample = SC_D_sample.clone()
        SC_D_noised_mask = batch['SC_D_mask'].clone()
        chi_1pi_periodic_mask = batch['chi_1pi_periodic_mask'].clone()

        metric = {}
        for i in range(self.NUM_CHI_ANGLES):
            SC_D_true_ = SC_D_true[..., i]
            SC_D_sample_ = SC_D_sample[..., i]
            chi_num = 1 if SC_D_noised_mask[..., i].sum() == 0 else SC_D_noised_mask[..., i].sum()

            chi_1pi_periodic_mask_ = chi_1pi_periodic_mask[..., i]

            chi_diff = (SC_D_sample_ - SC_D_true_).abs()

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

        predict_coords = get_atom14_coords(batch.X, batch.residue_type, batch.BB_D, SC_D_sample)
        metric["atom_rmsd"] = self.compute_rmsd(batch.X, predict_coords, batch.atom_mask, batch.residue_mask)
        return metric

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



