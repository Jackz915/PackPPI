## Install guide

### Install dependencies

``` bash
# clone project
git clone https://github.com/Jackz915/PackPPI
cd PackPPI

# create conda environment
conda env create -f environment.yaml
conda activate PackPPI

# install pytorch and cudatoolkit based on your CUDA version
conda install pytorch=2.0.1  cudatoolkit=11.7 -c pytorch

# pip install torch_cluster and torch_scatter in py3.9
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.3%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl

pip install torch_cluster-1.6.3+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt20cu117-cp39-cp39-linux_x86_64.whl
```


### Download dataset (optional)

``` bash
cd data

# PDBbind v.2020 and 3Dcomplex QS40
python download_pdbs.py

# Skempi_v2
bash download_skempi_v2.sh
```


