import os
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils.protein import from_pdb_file
from src.datamodules.components.helper import *


class ComplexDataset(Dataset):
    def __init__(self,
                 task: str = 'complex',
                 pdb_source: str = "rc",
                 path: Path = None,
                 entries: Dict = None,
                 model_data_cache_dir: Path = None,
                 cache_processed_data: bool = True,
                 force_process_data: bool = False
                 ) -> None:

        super().__init__()

        self.task = task
        self.pdb_source = pdb_source
        self.path = path
        self.entries = entries

        self.model_data_cache_dir = model_data_cache_dir
        self.cache_processed_data = cache_processed_data
        self.force_process_data = force_process_data

        os.makedirs(self.model_data_cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Data:
        pdb_code = self.entries[index]

        protein = {'pdb_code': pdb_code}
        # Load the PDB structure.
        protein.update(vars(from_pdb_file(Path(self.path,
                                               f'{pdb_code}_' + self.pdb_source + '.pdb'), mse_to_met=True)))

        return self.transform(protein)

    def transform(self, protein: Dict) -> Data:
        single_cache_path = Path(self.model_data_cache_dir, self.task, protein["pdb_code"])
        os.makedirs(single_cache_path, exist_ok=True)

        protein_data_filepath = Path(single_cache_path, 'protein_data.pt')
        protein_data = (
            torch.load(protein_data_filepath)
            if os.path.exists(str(protein_data_filepath))
            else None)

        data = self.prot_to_data(protein=protein,
                                 protein_data=protein_data,
                                 protein_data_filepath=protein_data_filepath,
                                 cache_processed_data=self.cache_processed_data,
                                 force_process_data=self.force_process_data)

        return data

    @staticmethod
    def prot_to_data(protein: Dict,
                     protein_data: Optional[Any] = None,
                     protein_data_filepath: Optional[Path] = None,
                     cache_processed_data: bool = True,
                     force_process_data: bool = False
                     ) -> Data:

        if protein_data is None or force_process_data:

            # Create all necessary residue features.
            total_num_residues = len(protein["aaindex"])
            X = torch.from_numpy(protein["atom_positions"]).to(torch.float32)
            residue_type = torch.from_numpy(protein["aaindex"]).to(torch.int64)
            atom_mask = torch.from_numpy(protein["atom_mask"]).to(torch.float32)
            residue_index = torch.from_numpy(protein["residue_index"]).to(torch.int64)

            unique_chain = pd.unique(protein["chain_id"])
            chain_indices = pd.Categorical(protein["chain_id"], categories=unique_chain, ordered=False)
            chain_indices = pd.factorize(chain_indices)[0] + 1
            chain_indices = torch.from_numpy(chain_indices).to(torch.int64)

            if len(unique_chain) > 1:
                index_offset = 0

                for chain_idx in torch.unique(chain_indices)[:-1]:
                    index_offset += max(residue_index[chain_indices == chain_idx])
                    index_offset += 100
                    residue_index[chain_indices == chain_idx + 1] += index_offset

            residue_mask = torch.isfinite(X[:, :4].sum(dim=(-1, -2))).to(torch.float32)

            BB_D, BB_D_mask = calc_bb_dihedrals(X, residue_index)
            SC_D, SC_D_mask = calc_sc_dihedrals(X, residue_type)

            BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
            BB_D_sincos = BB_D_sincos * BB_D_mask[..., None]

            SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)
            SC_D_sincos = SC_D_sincos * SC_D_mask[..., None]

            chi_1pi_periodic_mask = torch.tensor(rc.chi_pi_periodic)[residue_type].to(torch.bool)
            chi_2pi_periodic_mask = ~chi_1pi_periodic_mask

            X = X * residue_mask[..., None, None]
            residue_type = residue_type * residue_mask
            atom_mask = atom_mask * residue_mask[..., None]
            residue_index = residue_index * residue_mask
            chain_indices = chain_indices * residue_mask
            BB_D = BB_D * residue_mask[..., None]
            BB_D_sincos = BB_D_sincos * residue_mask[..., None, None]
            BB_D_mask = BB_D_mask * residue_mask[..., None]
            SC_D = SC_D * residue_mask[..., None]
            SC_D_sincos = SC_D_sincos * residue_mask[..., None, None]
            SC_D_mask = SC_D_mask * residue_mask[..., None]
            chi_1pi_periodic_mask = chi_1pi_periodic_mask * residue_mask[..., None]
            chi_2pi_periodic_mask = chi_2pi_periodic_mask * residue_mask[..., None]

            # Create new batch dictionary
            protein_data = Data(
                num_nodes=total_num_residues,
                X=X.to(torch.float32),
                atom_mask=atom_mask.to(torch.float32),
                residue_type=residue_type.to(torch.int64),
                residue_mask=residue_mask.to(torch.float32),
                residue_index=residue_index.to(torch.int64),
                chain_indices=chain_indices.to(torch.int64),
                BB_D=BB_D.to(torch.float32),
                BB_D_sincos=BB_D_sincos.to(torch.float32),
                BB_D_mask=BB_D_mask.to(torch.float32),
                SC_D=SC_D.to(torch.float32),
                SC_D_sincos=SC_D_sincos.to(torch.float32),
                SC_D_mask=SC_D_mask.to(torch.float32),
                chi_1pi_periodic_mask=torch.logical_and(SC_D_mask, chi_1pi_periodic_mask),
                chi_2pi_periodic_mask=torch.logical_and(SC_D_mask, chi_2pi_periodic_mask),
            )

            # Remove any potential nans
            remove_nans = lambda x: torch.nan_to_num(x) if isinstance(x, torch.Tensor) else x
            protein_data = protein_data.apply(remove_nans)

            if cache_processed_data:
                torch.save(protein_data, str(protein_data_filepath))

        return protein_data



