wget https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv
wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz
tar -xzvf SKEMPI2_PDBs.tgz

mv PDBs skempi_pdbs
rm SKEMPI2_PDBs.tgz
