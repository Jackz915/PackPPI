import random
from pytorch_lightning import LightningDataModule
import pickle
from src.datamodules.components.complex_dataset import ComplexDataset
from src.utils.protein import from_pdb_file
import torch.nn.functional as F
import os
import torch
from tqdm import tqdm
from pathlib import Path

from typing import *
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class ComplexDataModule(LightningDataModule):
    def __init__(
        self,
        task: str = "complex",
        pdb_source: str = "rc",
        data_dir: Path = None,
        pdb_filename: str = None,
        model_data_cache_dir: str = "dataset_cache",
        cache_processed_data: bool = True,
        force_process_data: bool = False,

        len_region: Tuple[int, int] = None,
        data_split: Tuple[int, int, int] = None,

        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.split_file = Path(self.hparams.data_dir, "pretrain_" + self.hparams.task + ".pkl")

    def load_complex_entries(self) -> List:
        pdb_codes = list(map(lambda s: s[:4],
                             os.listdir(Path(self.hparams.data_dir, self.hparams.pdb_filename))))
        entries = []
        with tqdm(total=len(pdb_codes), desc='Filtering the pdb based on length range') as pbar:
            for pdb_code in pdb_codes:
                pdb_path = Path(self.hparams.data_dir,
                                self.hparams.pdb_filename,
                                f'{pdb_code}_' + self.hparams.pdb_source + '.pdb')
                protein = from_pdb_file(pdb_path, mse_to_met=True)
                pro_length = len(protein.aaindex)
                if pro_length < self.hparams.len_region[0] or pro_length > self.hparams.len_region[1]:
                    continue
                else:
                    entries.append(pdb_code)
                    pbar.update(1)
        return entries

    def random_split(self, entries: List) -> Dict:
        random.shuffle(entries)
        test_count = int(self.hparams.data_split[2] * len(entries))
        test_clusters = entries[:test_count]
        valid_count = int(self.hparams.data_split[1] * len(entries))
        valid_clusters = entries[test_count:(test_count + valid_count)]
        train_clusters = entries[(test_count + valid_count):]

        data_splits = {
            'train': train_clusters,
            'valid': valid_clusters,
            'test': test_clusters
        }

        with open(self.split_file, 'wb') as f:
            pickle.dump(data_splits, f)
        return data_splits

    def setup(self, stage: Optional[str] = None):
        if not os.path.exists(self.split_file):
            entries = self.load_complex_entries()
            data_splits = self.random_split(entries)
        else:
            with open(self.split_file, 'rb') as f:
                data_splits = pickle.load(f)

        train_entries = data_splits['train']
        valid_entries = data_splits['valid']
        test_entries = data_splits['test']

        # predict_pdbs = os.listdir(self.hparams.inference_dir)
        #
        # if stage in ["predict"]:
        #     assert len(predict_pdbs) > 0, "PDB inputs must be provided during model inference."

        self.train_set = ComplexDataset(
            task=self.hparams.task,
            pdb_source=self.hparams.pdb_source,
            path=Path(self.hparams.data_dir, self.hparams.pdb_filename),
            entries=train_entries,
            model_data_cache_dir=Path(self.hparams.data_dir, self.hparams.model_data_cache_dir),
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        )

        self.val_set = ComplexDataset(
            task=self.hparams.task,
            pdb_source=self.hparams.pdb_source,
            path=Path(self.hparams.data_dir, self.hparams.pdb_filename),
            entries=valid_entries,
            model_data_cache_dir=Path(self.hparams.data_dir, self.hparams.model_data_cache_dir),
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        )

        self.test_set = ComplexDataset(
            task=self.hparams.task,
            pdb_source=self.hparams.pdb_source,
            path=Path(self.hparams.data_dir, self.hparams.pdb_filename),
            entries=test_entries,
            model_data_cache_dir=Path(self.hparams.data_dir, self.hparams.model_data_cache_dir),
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        )

        # self.predict_set = ComplexDataset(
        #     path=self.hparams.inference_dir,
        #     entries=predict_entries,
        #     model_data_cache_dir=Path(self.hparams.inference_dir, self.hparams.model_data_cache_dir),
        # )

    def get_dataloader(
            self,
            dataset: ComplexDataset,
            batch_size: int,
            pin_memory: bool,
            shuffle: bool,
            drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return self.get_dataloader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    # def predict_dataloader(self):
    #     return self.get_dataloader(
    #         self.predict_set,
    #         batch_size=self.hparams.predict_batch_size,
    #         pin_memory=self.hparams.predict_pin_memory,
    #         shuffle=False,
    #         drop_last=False,
    #     )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def collate_fn(protein_batch):
    # Padding function
    max_size = max([protein.num_nodes for protein in protein_batch])

    def _maybe_pad(protein, attr):
        attr_tensor = getattr(protein, attr)
        return F.pad(attr_tensor, [0, 0] * (len(attr_tensor.shape) - 1) + [0, max_size - protein.num_nodes])

    # Create batch by stacking all features
    batch = Data(
        num_proteins=len(protein_batch),
        max_size=max_size,
        X=torch.stack([_maybe_pad(protein, "X") for protein in protein_batch]),  # [B, L, 14, 3]
        atom_mask=torch.stack([_maybe_pad(protein, "atom_mask") for protein in protein_batch]),  # [B, L, 14]
        chain_indices=torch.stack([_maybe_pad(protein, "chain_indices") for protein in protein_batch]),  # [B, L]
        residue_mask=torch.stack([_maybe_pad(protein, "residue_mask") for protein in protein_batch]),  # [B, L]
        residue_index=torch.stack([_maybe_pad(protein, "residue_index") for protein in protein_batch]),  # [B, L]
        residue_type=torch.stack([_maybe_pad(protein, "residue_type") for protein in protein_batch]),  # [B, L]
        BB_D=torch.stack([_maybe_pad(protein, "BB_D") for protein in protein_batch]),  # [B, L, 3]
        BB_D_sincos=torch.stack([_maybe_pad(protein, "BB_D_sincos") for protein in protein_batch]),  # [B, L, 3, 2]
        BB_D_mask=torch.stack([_maybe_pad(protein, "BB_D_mask") for protein in protein_batch]),  # [B, L, 3]
        SC_D=torch.stack([_maybe_pad(protein, "SC_D") for protein in protein_batch]),  # [B, L, 4]
        SC_D_sincos=torch.stack([_maybe_pad(protein, "SC_D_sincos") for protein in protein_batch]),  # [B, L, 4, 2]
        SC_D_mask=torch.stack([_maybe_pad(protein, "SC_D_mask") for protein in protein_batch]),  # [B, L, 4]
        chi_1pi_periodic_mask=torch.stack([_maybe_pad(protein, "chi_1pi_periodic_mask") for protein in protein_batch]),
        # [B, L, 4]
        chi_2pi_periodic_mask=torch.stack([_maybe_pad(protein, "chi_2pi_periodic_mask") for protein in protein_batch]),
        # [B, L, 4]
    )

    return batch


