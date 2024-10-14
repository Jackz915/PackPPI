import os
import numpy as np
import random
import pickle
import pandas as pd
import math
from pytorch_lightning import LightningDataModule
from pathlib import Path

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from src.datamodules.components.skempi_dataset import SkempiDataset


class SkempiDataModule(LightningDataModule):
    def __init__(
            self,
            task: str = 'skempi',
            data_dir: Path = None,
            pdb_filename: str = None,
            meta_filename: str = 'skempi_v2.csv',
            block_list: Optional[List] = None,
            model_data_cache_dir: str = "dataset_cache",

            use_esm: bool = False,
            num_cvfolds: int = 3,
            cvfold_index: int = 0,
            split_seed: int = 42,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,

            cache_processed_data: bool = True,
            force_process_data: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # features - ESM protein sequence embeddings
        if self.hparams.use_esm:
            self.esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
            self.esm_model = self.esm_model.eval().cpu()
            self.esm_batch_converter = esm_alphabet.get_batch_converter()

        self.split_file = Path(self.hparams.data_dir, "pretrain_" + self.hparams.task + ".pkl")

    def load_skempi_entries(self, meta_filename, pdb_filename, block_list):
        df = pd.read_csv(Path(self.hparams.data_dir, meta_filename), sep=';')
        df['dG_wt'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
        df['dG_mut'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
        df['ddG'] = df['dG_mut'] - df['dG_wt']

        def _parse_mut(mut_name):
            wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
            mutseq = int(mut_name[2:-1])
            return {
                'wt': wt_type,
                'mt': mt_type,
                'chain': mutchain,
                'resseq': mutseq,
                'icode': ' ',
                'name': mut_name
            }

        entries = []
        for i, row in df.iterrows():
            pdbcode, group1, group2 = row['#Pdb'].split('_')
            if pdbcode in block_list:
                continue
            mut_str = row['Mutation(s)_cleaned']
            muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
            if muts[0]['chain'] in group1:
                group_ligand, group_receptor = group1, group2
            else:
                group_ligand, group_receptor = group2, group1

            pdb_path = Path(self.hparams.data_dir, pdb_filename, '{}.pdb'.format(pdbcode.upper()))
            if not os.path.exists(pdb_path):
                continue

            if not np.isfinite(row['ddG']):
                continue

            entry = {
                'id': i,
                'complex': row['#Pdb'],
                'mutstr': mut_str,
                'num_muts': len(muts),
                'pdb_id': pdbcode,
                'group_ligand': list(group_ligand),
                'group_receptor': list(group_receptor),
                'mutations': muts,
                'ddG': np.float32(row['ddG']),
                'pdb_path': pdb_path,
            }
            entries.append(entry)

        return entries

    def _split(self, entries: List[Dict]) -> Dict:
        complex_to_entries = {}
        for e in entries:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        random.Random(self.hparams.split_seed).shuffle(complex_list)

        split_size = math.ceil(len(complex_list) / self.hparams.num_cvfolds)
        complex_splits = [
            complex_list[i * split_size: (i + 1) * split_size]
            for i in range(self.hparams.num_cvfolds)
        ]

        val_split = complex_splits.pop(self.hparams.cvfold_index)
        train_split = sum(complex_splits, start=[])

        train_entries, valid_entries = [], []
        for cplx in train_split:
            train_entries += complex_to_entries[cplx]

        for cplx in val_split:
            valid_entries += complex_to_entries[cplx]

        data_splits = {
            'train': train_entries,
            'valid': valid_entries,
        }
        
        with open(self.split_file, 'wb') as f:
           pickle.dump(data_splits, f)

        return data_splits

    def setup(self, stage: Optional[str] = None):

        # Create split file (if needed) and load it
        if not os.path.exists(self.split_file) or self.hparams.force_process_data:
            entries = self.load_skempi_entries(self.hparams.meta_filename,
                                               self.hparams.pdb_filename,
                                               self.hparams.block_list)

            data_splits = self._split(entries)
        else:
            with open(self.split_file, 'rb') as f:
                data_splits = pickle.load(f)

        train_entries = data_splits['train']
        valid_entries = data_splits['valid']

        self.train_set = SkempiDataset(
            path=Path(self.hparams.data_dir, self.hparams.pdb_filename),
            entries=train_entries,
            model_data_cache_dir=Path(self.hparams.data_dir, self.hparams.model_data_cache_dir),
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data,
        )

        self.val_set = SkempiDataset(
            path=Path(self.hparams.data_dir, self.hparams.pdb_filename),
            entries=valid_entries,
            model_data_cache_dir=Path(self.hparams.data_dir, self.hparams.model_data_cache_dir),
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data,
        )

    def get_dataloader(
            self,
            dataset: SkempiDataset,
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
            collate_fn=collate_fn_esm if self.hparams.use_esm else collate_fn
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

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

def collate_fn_esm(protein_batch):
    # Padding function
    max_size = max([protein.num_nodes for protein in protein_batch])

    def _maybe_pad(protein, attr):
        attr_tensor = getattr(protein, attr)
        return F.pad(attr_tensor, [0, 0] * (len(attr_tensor.shape) - 1) + [0, max_size - protein.num_nodes])

    # Create batch by stacking all features
    batch = Data(
        num_proteins=len(protein_batch),
        max_size=max_size,
        ddg=torch.stack([getattr(protein, "ddg") for protein in protein_batch]),  # [B, L, 14, 3]
        esm_representations=torch.stack([_maybe_pad(protein, "esm_representations") for protein in protein_batch]),
        esm_representations_mut=torch.stack([_maybe_pad(protein, "esm_representations_mut") for protein in protein_batch]),
    )

    return batch


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
        ddg=torch.stack([getattr(protein, "ddg") for protein in protein_batch]),  # [B]
        mut_mask=torch.stack([_maybe_pad(protein, "mut_mask") for protein in protein_batch]),  # [B, L]

        # common feature
        X=torch.stack([_maybe_pad(protein, "X") for protein in protein_batch]),  # [B, L, 14, 3]
        residue_mask=torch.stack([_maybe_pad(protein, "residue_mask") for protein in protein_batch]),  # [B, L]
        residue_index=torch.stack([_maybe_pad(protein, "residue_index") for protein in protein_batch]),  # [B, L]
        chain_indices=torch.stack([_maybe_pad(protein, "chain_indices") for protein in protein_batch]),  # [B, L]
        BB_D=torch.stack([_maybe_pad(protein, "BB_D") for protein in protein_batch]),  # [B, L, 3]
        BB_D_sincos=torch.stack([_maybe_pad(protein, "BB_D_sincos") for protein in protein_batch]),  # [B, L, 3, 2]
        BB_D_mask=torch.stack([_maybe_pad(protein, "BB_D_mask") for protein in protein_batch]),  # [B, L, 3]

        # wild feature
        atom_mask=torch.stack([_maybe_pad(protein, "atom_mask") for protein in protein_batch]),  # [B, L, 14]
        residue_type=torch.stack([_maybe_pad(protein, "residue_type") for protein in protein_batch]),  # [B, L]
        SC_D=torch.stack([_maybe_pad(protein, "SC_D") for protein in protein_batch]),  # [B, L, 3]
        SC_D_sincos=torch.stack([_maybe_pad(protein, "SC_D_sincos") for protein in protein_batch]),  # [B, L, 4, 2]
        SC_D_mask=torch.stack([_maybe_pad(protein, "SC_D_mask") for protein in protein_batch]),  # [B, L, 4]
        chi_1pi_periodic_mask=torch.stack([_maybe_pad(protein, "chi_1pi_periodic_mask") for protein in protein_batch]),  # [B, L, 4]
        chi_2pi_periodic_mask=torch.stack([_maybe_pad(protein, "chi_2pi_periodic_mask") for protein in protein_batch]),  # [B, L, 4]

        # mutation feature
        atom_mask_mut=torch.stack([_maybe_pad(protein, "atom_mask_mut") for protein in protein_batch]),  # [B, L, 14]
        residue_type_mut=torch.stack([_maybe_pad(protein, "residue_type_mut") for protein in protein_batch]),  # [B, L]
        SC_D_mut=torch.stack([_maybe_pad(protein, "SC_D_mut") for protein in protein_batch]),  # [B, L, 3]
        SC_D_sincos_mut=torch.stack([_maybe_pad(protein, "SC_D_sincos_mut") for protein in protein_batch]),  # [B, L, 4, 2]
        SC_D_mask_mut=torch.stack([_maybe_pad(protein, "SC_D_mask_mut") for protein in protein_batch]),  # [B, L, 4]
        chi_1pi_periodic_mask_mut=torch.stack([_maybe_pad(protein, "chi_1pi_periodic_mask_mut") for protein in protein_batch]),   # [B, L, 4]
        chi_2pi_periodic_mask_mut=torch.stack([_maybe_pad(protein, "chi_2pi_periodic_mask_mut") for protein in protein_batch]),   # [B, L, 4]
    )

    return batch
