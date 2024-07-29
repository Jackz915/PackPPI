<div align="center">

# PackPPI
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

A integrated framework for protein-protein complex side-chain packing and <br> 
ΔΔG prediction based on diffusion model.
</div>

## Overview :mag:
The PackPPI framework comprises three functional modules: side-chain conformation modeling of protein complexes (PackPPI-MSC), proximal optimization (PackPPI-Prox), and prediction of the effect of mutations on binding affinity (PackPPI-AP). Given the structural context of a protein complex, the framework first employs a joint diffusion probabilistic model to generate reliable side-chain conformations. It then uses the proximal gradient descent method to avoid atomic collisions during sampling, obtaining high-confidence target structures. Subsequently, the geometric features of the learned structural context are utilized for downstream ΔΔG prediction.

![framework.png](./imgs/Framework.png)


## Install guide :rocket:

### Install dependencies
``` bash
# 1. clone project
git clone https://github.com/Jackz915/PackPPI

# 2. create conda environment
cd PackPPI
conda env create -f environment.yaml
conda activate PackPPI

# 3. install pytorch and cudatoolkit based on your CUDA version
conda install pytorch=2.0.1  cudatoolkit=11.7 -c pytorch

# 4. pip install torch_cluster and torch_scatter in py3.9
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.3%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl

pip install torch_cluster-1.6.3+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt20cu117-cp39-cp39-linux_x86_64.whl

# 5. MolProbity Installation
# Please follow the recommended protocol: https://github.com/rlabduke/MolProbity
# The molprobity.clashscore function will returns a list of atoms with impossible steric clashes and the clashscore.
```

### Download trained models
https://drive.google.com/drive/folders/1MbvDKjQJIMafll5Sy3ZI2rJaLMOj8CkT?usp=sharing  <br>
**Note**: Make sure you have specified the checkpoint path in the `ckpt_path` parameter in configuration.

### Download dataset (optional)
``` bash
cd data

# PDBbind v.2020 and 3Dcomplex QS40
python download_pdbs.py

# Skempi_v2
bash download_skempi_v2.sh
```


## Usage :sparkles:

### Side-chain conformation modeling of protein complexes (PackPPI-MSC)
- Train model with default configuration:

``` bash
python src/train_diffusion.py
```

- Evaluate model (change the `ckpt_path` in configs/eval_diffusion.yaml to your trained model path):
``` bash
python src/eval_diffusion.py --h

"""
usage: eval_diffusion.py [-h] --input INPUT --outdir OUTDIR --molprobity_clash_loc MOLPROBITY_CLASH_LOC [--use_proximal] [--device DEVICE]
optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         The input pdb file path.
  --outdir OUTDIR       Directory to store outputs.
  --molprobity_clash_loc MOLPROBITY_CLASH_LOC
                        Path to /build/bin/molprobity.clashscore.
  --use_proximal        Use proximal optimize.
  --device DEVICE       cuda or cpu.
"""
```

- Example:
``` bash
python src/eval_diffusion.py --input data/6ere.pdb \
                             --outdir temp \
                             --molprobity_clash_loc ~/MolProbity/build/bin/molprobity.clashscore \
                             --device cuda

# Output
{'chi_0_ae_rad': tensor(0.2468), 'chi_0_ae_deg': tensor(14.1410), 'chi_0_acc': tensor(0.7935),
'chi_1_ae_rad': tensor(0.3722), 'chi_1_ae_deg': tensor(21.3241), 'chi_1_acc': tensor(0.5446),
'chi_2_ae_rad': tensor(0.7627), 'chi_2_ae_deg': tensor(43.7001), 'chi_2_acc': tensor(0.3306),
'chi_3_ae_rad': tensor(1.0014), 'chi_3_ae_deg': tensor(57.3753), 'chi_3_acc': tensor(0.2222),
'atom_rmsd': tensor(0.8681), 'total_acc': tensor(0.4727), 'interface_acc': tensor(0.4035),
'clashscore': 19.47}
```

``` bash
python src/eval_diffusion.py --input data/6ere.pdb \
                             --outdir temp \
                             --molprobity_clash_loc ~/MolProbity/build/bin/molprobity.clashscore \
                             --device cuda \
                             --use_proximal

# Output
{'chi_0_ae_rad': tensor(0.2622), 'chi_0_ae_deg': tensor(15.0255), 'chi_0_acc': tensor(0.8071),
'chi_1_ae_rad': tensor(0.3870), 'chi_1_ae_deg': tensor(22.1737), 'chi_1_acc': tensor(0.5287),
'chi_2_ae_rad': tensor(0.7673), 'chi_2_ae_deg': tensor(43.9655), 'chi_2_acc': tensor(0.3967),
'chi_3_ae_rad': tensor(0.9359), 'chi_3_ae_deg': tensor(53.6242), 'chi_3_acc': tensor(0.2222),
'atom_rmsd': tensor(0.9315), 'total_acc': tensor(0.4887), 'interface_acc': tensor(0.5389),
'clashscore': 14.86}
```

The output structure is in the outdir folder named `structure.pdb` <br>
**Note**: If you are using the **--use_proximal** parameter and running on a GPU, ensure that you have sufficient memory space.
For larger protein structures (e.g., larger than **1500** amino acids), you may encounter memory insufficiency errors.
In such cases, consider switching to CPU, although this will increase the runtime.


### Prediction of mutation effect on binding affinity (PackPPI-AP)
- Train model with default configuration:

``` bash
# PackPPI-AP with mutation encoder:
python src/train_affinity.py experiment=affinity_network.yaml

# PackPPI-AP without mutation encoder:
python src/train_affinity.py experiment=affinity_linear.yaml

# ESM-2 pretrained model (esm2_t33_650M_UR50D):
# make sure you install esmfold in your conda environment. (https://github.com/facebookresearch/esm) 
python src/train_affinity.py experiment=affinity_esm.yaml
```

- Evaluate model (change the `ckpt_path` in configs/eval_affinity.yaml to your trained model path):
``` bash
python src/eval_affinity.py -h

"""
usage: eval_affinity.py [-h] --input INPUT --mutstr MUTSTR [--device DEVICE]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    The input pdb file path.
  --mutstr MUTSTR  A string containing wild-type residue, chain ID, position, and mutant residue (e.g., "RA47A").
                   If more than one mutation, please separated by commas (e.g., "RA47A,EA48A").
  --device DEVICE  cuda or cpu
"""
```

- Example (Single mutation):
``` bash
python src/eval_affinity.py --input data/2FTL.pdb \
                            --mutstr KI15G \
                            --device cuda

# Output
----- The predicted binding affinity change (wildtype-mutant) is 13.1932 kcal/mol -----
```

- Example (Multi mutations):
``` bash
python src/eval_affinity.py --input data/1BRS.pdb \
                            --mutstr KA25A,DD35A \
                            --device cuda

# Output
----- The predicted binding affinity change (wildtype-mutant) is 8.2439 kcal/mol -----
```







