import os
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils.protein import from_pdb_file
from src.datamodules.components.helper import *


class SkempiDataset(Dataset):
    def __init__(self,
                 task: str = 'skempi',
                 path: Path = None,
                 entries: Dict = None,

                 esm_model: Optional[Any] = None,
                 esm_batch_converter: Optional[Any] = None,

                 model_data_cache_dir: Path = None,
                 cache_processed_data: bool = True,
                 force_process_data: bool = False,
                 ) -> None:

        super().__init__()

        self.task = task
        self.path = path
        self.entries = entries

        self.esm_model = esm_model
        self.esm_batch_converter = esm_batch_converter

        self.model_data_cache_dir = model_data_cache_dir
        self.cache_processed_data = cache_processed_data
        self.force_process_data = force_process_data

        os.makedirs(self.model_data_cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Data:

        protein = self.entries[index]
        protein["pdb_id"] = protein["pdb_id"].upper()

        # Load the PDB structure.
        protein.update(vars(from_pdb_file(protein["pdb_path"], mse_to_met=True)))

        return self.transform(protein)

    def transform(self, protein: Dict) -> Data:
        single_cache_path = Path(self.model_data_cache_dir, self.task, protein["pdb_id"] + str(protein["id"]))
        os.makedirs(single_cache_path, exist_ok=True)

        protein_data_filepath = Path(single_cache_path, 'protein_data.pt')
        protein_data = (
            torch.load(protein_data_filepath)
            if os.path.exists(str(protein_data_filepath))
            else None)

        data = self.prot_to_data(protein,
                                 protein_data=protein_data,
                                 protein_data_filepath=protein_data_filepath,
                                 cache_path=single_cache_path,
                                 esm_model=self.esm_model,
                                 esm_batch_converter=self.esm_batch_converter,
                                 cache_processed_data=self.cache_processed_data,
                                 force_process_data=self.force_process_data)

        return data

    @staticmethod
    def prot_to_data(protein: Dict,
                     protein_data: Optional[Any] = None,
                     protein_data_filepath: Optional[Path] = None,
                     cache_path: Optional[Path] = None,
                     esm_model: Optional[nn.Module] = None,
                     esm_batch_converter: Optional[Any] = None,
                     cache_processed_data: bool = True,
                     force_process_data: bool = False
                     ) -> Data:

        if protein_data is None or force_process_data:

            # Create all necessary features.
            num_nodes = protein["atom_positions"].shape[0]
            X = torch.from_numpy(protein["atom_positions"]).to(torch.float32)
            residue_type = torch.from_numpy(protein["aaindex"]).to(torch.int64)
            atom_mask = torch.from_numpy(protein["atom_mask"]).to(torch.float32)
            residue_index = torch.from_numpy(protein["residue_index"]).to(torch.int64)
            chain_id = protein["chain_id"]

            if "ddG" in protein:
                ddg = torch.tensor(protein["ddG"]).to(torch.float32)
            else:
                ddg = torch.tensor(0.0, dtype=torch.float32)

            # Create chain_indices
            unique_chain = pd.unique(chain_id)
            chain_indices = pd.Categorical(chain_id, categories=unique_chain, ordered=False)
            chain_indices = pd.factorize(chain_indices)[0] + 1
            chain_indices = torch.from_numpy(chain_indices).to(torch.int64)

            residue_mask = torch.isfinite(torch.sum(X[:, :4], dim=(-1, -2))).to(torch.float32)
            
            # Create wild residue features.
            BB_D, BB_D_mask = calc_bb_dihedrals(X, residue_index)
            SC_D, SC_D_mask = calc_sc_dihedrals(X, residue_type)

            BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
            BB_D_sincos = BB_D_sincos * BB_D_mask[..., None]

            SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)
            SC_D_sincos = SC_D_sincos * SC_D_mask[..., None]

            chi_1pi_periodic_mask = torch.tensor(rc.chi_pi_periodic)[residue_type].to(torch.bool)
            chi_2pi_periodic_mask = ~chi_1pi_periodic_mask

            # Create mutation residue features.
            residue_type_mut, atom_mask_mut, SC_D_mut, SC_D_sincos_mut = residue_type.clone(), atom_mask.clone(), SC_D.clone(), SC_D_sincos.clone()
            for mut in protein['mutations']:
                mut_chain = mut['chain']
                mut_resseq = mut['resseq']
                mut_mt = mut['mt']
                mut_wt = mut['wt']
                if mut_chain in chain_id and mut_mt in rc.restypes:
                    # get mutation idx
                    index = np.logical_and(chain_id == mut_chain, residue_index == mut_resseq).to(torch.bool)

                    # change residue_type
                    ref_wt = rc.restypes[residue_type_mut[index]] 
                    
                    if ref_wt != mut_wt:
                        raise ValueError(f"The mutation: {mut_wt}{mut_chain}{mut_resseq}{mut_mt} is inconsistent with wild-type {ref_wt} in PDB file")
                    else:
                        residue_type_mut[index] = rc.restypes.index(mut_mt)

                    # change atom mask
                    atoms = rc.restype_name_to_atom14_names[rc.restype_1to3[mut_mt]]
                    mask = [1. if atom else 0. for atom in atoms]
                    atom_mask_mut[index] = torch.tensor(mask, dtype=torch.float32)

                    # change SC_D
                    SC_D_mut[index] = 0.
                    SC_D_sincos_mut[index] = 0.
                else:
                    print(f"Ignore the mutation: {mut_wt}{mut_chain}{mut_resseq}{mut_mt}")
                    continue
                    
            if esm_model is not None:
                esm_representations, esm_representations_mut = torch.zeros_like(residue_type), torch.zeros_like(residue_type)
                esm_representations = get_esm_feature(sequence=np.array(rc.restypes_with_x)[residue_type],
                                                      chain=chain_indices,
                                                      residues_mask=None,
                                                      padding_length=20,
                                                      esm_model=esm_model,
                                                      esm_batch_converter=esm_batch_converter)
                esm_representations = esm_representations * residue_mask[..., None]

                esm_representations_mut = get_esm_feature(sequence=np.array(rc.restypes_with_x)[residue_type_mut],
                                                          chain=chain_indices,
                                                          residues_mask=None,
                                                          padding_length=20,
                                                          esm_model=esm_model,
                                                          esm_batch_converter=esm_batch_converter)
                esm_representations_mut = esm_representations_mut * residue_mask[..., None]
                
                # Create new batch dictionary
                protein_data = Data(
                    num_nodes=num_nodes,
                    ddg=ddg.to(torch.float32),
                    esm_representations=esm_representations.to(torch.float32),
                    esm_representations_mut=esm_representations_mut.to(torch.float32),
                )
    
                # Remove any potential nans
                remove_nans = lambda x: torch.nan_to_num(x) if isinstance(x, torch.Tensor) else x
                protein_data = protein_data.apply(remove_nans)
                
            else:   
                # Add chain offset to residue_index
                if len(unique_chain) > 1:
                    index_offset = 0
    
                    for chain_idx in torch.unique(chain_indices)[:-1]:
                        index_offset += max(residue_index[chain_indices == chain_idx])
                        index_offset += 100
                        residue_index[chain_indices == chain_idx + 1] += index_offset
                
                _, SC_D_mask_mut = calc_sc_dihedrals(X, residue_type_mut)
    
                chi_1pi_periodic_mask_mut = torch.tensor(rc.chi_pi_periodic)[residue_type_mut].to(torch.bool)
                chi_2pi_periodic_mask_mut = ~chi_1pi_periodic_mask_mut
    
                mut_mask = (residue_type != residue_type_mut).to(torch.int64)
    
                X = X * residue_mask[..., None, None]
                mut_mask = mut_mask * residue_mask
                residue_type = residue_type * residue_mask
                residue_type_mut = residue_type_mut * residue_mask
                atom_mask = atom_mask * residue_mask[..., None]
                atom_mask_mut = atom_mask_mut * residue_mask[..., None]
                residue_index = residue_index * residue_mask
                chain_indices = chain_indices * residue_mask
                BB_D = BB_D * residue_mask[..., None]
                BB_D_sincos = BB_D_sincos * residue_mask[..., None, None]
                BB_D_mask = BB_D_mask * residue_mask[..., None]
                SC_D = SC_D * residue_mask[..., None]
                SC_D_sincos = SC_D_sincos * residue_mask[..., None, None]
                SC_D_mask = SC_D_mask * residue_mask[..., None]
                SC_D_mut = SC_D_mut * residue_mask[..., None]
                SC_D_sincos_mut = SC_D_sincos_mut * residue_mask[..., None, None]
                SC_D_mask_mut = SC_D_mask_mut * residue_mask[..., None]
                chi_1pi_periodic_mask = chi_1pi_periodic_mask * residue_mask[..., None]
                chi_2pi_periodic_mask = chi_2pi_periodic_mask * residue_mask[..., None]
                chi_1pi_periodic_mask_mut = chi_1pi_periodic_mask_mut * residue_mask[..., None]
                chi_2pi_periodic_mask_mut = chi_2pi_periodic_mask_mut * residue_mask[..., None]
    
                # Create new batch dictionary
                protein_data = Data(
                    num_nodes=num_nodes,
                    ddg=ddg.to(torch.float32),
                    mut_mask=mut_mask.to(torch.float32),
    
                    # common feature
                    X=X.to(torch.float32),
                    residue_mask=residue_mask.to(torch.float32),
                    residue_index=residue_index.to(torch.int64),
                    chain_indices=chain_indices.to(torch.int64),
                    BB_D=BB_D.to(torch.float32),
                    BB_D_sincos=BB_D_sincos.to(torch.float32),
                    BB_D_mask=BB_D_mask.to(torch.float32),
    
                    # wild feature
                    atom_mask=atom_mask.to(torch.float32),
                    residue_type=residue_type.to(torch.int64),
                    SC_D=SC_D.to(torch.float32),
                    SC_D_sincos=SC_D_sincos.to(torch.float32),
                    SC_D_mask=SC_D_mask.to(torch.float32),
                    chi_1pi_periodic_mask=torch.logical_and(SC_D_mask, chi_1pi_periodic_mask),
                    chi_2pi_periodic_mask=torch.logical_and(SC_D_mask, chi_2pi_periodic_mask),
    
                    # mutation feature
                    atom_mask_mut=atom_mask_mut.to(torch.float32),
                    residue_type_mut=residue_type_mut.to(torch.int64),
                    SC_D_mut=SC_D_mut.to(torch.float32),
                    SC_D_sincos_mut=SC_D_sincos_mut.to(torch.float32),
                    SC_D_mask_mut=SC_D_mask_mut.to(torch.float32),
                    chi_1pi_periodic_mask_mut=torch.logical_and(SC_D_mask_mut, chi_1pi_periodic_mask_mut),
                    chi_2pi_periodic_mask_mut=torch.logical_and(SC_D_mask_mut, chi_2pi_periodic_mask_mut),
                )
    
                # Remove any potential nans
                remove_nans = lambda x: torch.nan_to_num(x) if isinstance(x, torch.Tensor) else x
                protein_data = protein_data.apply(remove_nans)

            if cache_processed_data:
                torch.save(protein_data, str(protein_data_filepath))

        return protein_data

