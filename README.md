<div align="left">

# PackPPI
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

![framework.png](./imgs/framework.png)

</div>



## Install guide

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


## Usage

### Side-chain conformation modeling of protein complexes (PackPPI-MSC)
- Train model with default configuration:

``` bash
python src/train_diffusion.py
```

- Evaluate model:
``` bash
# Run
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

**Note**: If you are using the --use_proximal parameter and running on a GPU, ensure that you have sufficient memory space.
For larger protein structures (e.g., greater than 1500 amino acids), you may encounter memory insufficiency errors.
In such cases, consider switching to CPU, although this will increase the runtime.
"""

# Example output:
{'chi_0_ae_rad': tensor(0.2646), 'chi_0_ae_deg': tensor(15.1623), 'chi_0_acc': tensor(0.7979),
'chi_1_ae_rad': tensor(0.3080), 'chi_1_ae_deg': tensor(17.6488), 'chi_1_acc': tensor(0.6268),
'chi_2_ae_rad': tensor(0.8465), 'chi_2_ae_deg': tensor(48.5010), 'chi_2_acc': tensor(0.3008),
'chi_3_ae_rad': tensor(0.8517), 'chi_3_ae_deg': tensor(48.7990), 'chi_3_acc': tensor(0.2581),
'atom_rmsd': tensor(0.7784),
'total_acc': tensor(0.4959),
'interface_acc': tensor(0.5438),
'clashscore': 11.97}

# The output structure is in the outdir folder named `sample.pdb`
```


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

- Evaluate model:
``` bash
python src/eval_affinity.py (Coming soon)
```






