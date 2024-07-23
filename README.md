## Install guide

### Install dependencies

``` bash
# 1. clone project
git clone https://github.com/Jackz915/PackPPI
cd PackPPI

# 2. create conda environment
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
Train model with default configuration:

``` bash
python src/train.py train_diffusion.py
```

Inference:
``` bash
python src/inference_diffusion.py --h

"""
usage: inference_diffusion.py [-h] --input INPUT --outdir OUTDIR --molprobity_clash_loc MOLPROBITY_CLASH_LOC [--use_proximal] [--device DEVICE]
optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         The input pdb file path.
  --outdir OUTDIR       Directory to store outputs.
  --molprobity_clash_loc MOLPROBITY_CLASH_LOC
                        Path to /build/bin/molprobity.clashscore.
  --use_proximal        Use proximal optimize.
  --device DEVICE
"""
```

python src/inference_diffusion.py --input path/to/your/pdb_file \
                                  --outdir tem1 \
                                  --molprobity_clash_loc /home/zhangjk/MolProbity/build/bin/molprobity.clashscore \
                                  --use_proximal \
                                  --device 'cuda'
```




