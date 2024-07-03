import os
import io
import argparse
import requests
import gzip
import shutil
import tarfile
import zipfile
import tempfile
from requests import HTTPError
from pathlib import Path
from tqdm import tqdm
import concurrent.futures


PDB_RCSB = "rc"
PDB_RCSB_DOWNLOAD_URL_TEMPLATE = r"https://files.rcsb.org/download/{pdb_code}.pdb.gz"
PDB_REDO = "re"
PDB_REDO_DOWNLOAD_URL_TEMPLATE = "https://pdb-redo.eu/db/{pdb_code}/{pdb_code}_final.pdb"

PDB_DOWNLOAD_SOURCES = {
    PDB_RCSB: PDB_RCSB_DOWNLOAD_URL_TEMPLATE,
    PDB_REDO: PDB_REDO_DOWNLOAD_URL_TEMPLATE,
}


class PdbCodeFetcher:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

        # pdb-code for pre-training
        self.pdbbind_index_url = 'http://www.pdbbind.org.cn/download/PDBbind_v2020_plain_text_index.tar.gz'
        self.complex_QS40_url = 'https://shmoo.weizmann.ac.il/elevy/3dcomplexV6/dataV6/files/NRX_0_5_40_topo_label_compo.txt'
        self.PP_complex_file = 'INDEX_general_PP.2020'

        # pdb-code for fine-tuned
        self.skempi_url = 'https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv'

        # ecod-based complex screening based on T/F group
        self.ecod_url = 'https://raw.githubusercontent.com/rcsb/py-rcsb_exdb_assets/master/fall_back/ECOD/ecod.latest.domains.txt.gz'
        self.ecod_tfgroup_dict = self.load_ecod_dict()

    def get_final_code(self):
        pdbbind_codes, complex_codes, fine_codes = (set(self.get_pdbbind_code()),
                                                    set(self.get_3dcomplex_code()),
                                                    set(self.get_skempi_code()))
        merged_set = set()
        for lst in [self.get_tfgroup(fine_code) for fine_code in fine_codes]:
            merged_set.update(lst)

        pre_codes = list((pdbbind_codes | complex_codes) - fine_codes)

        final_codes = []
        for pre_code in pre_codes:
            if not self.get_tfgroup(pre_code):
                continue
            else:
                if not set(self.get_tfgroup(pre_code)) & merged_set:
                    final_codes.append(pre_code)
                else:
                    continue
        return final_codes

    def get_tfgroup(self, pdb_code):
        try:
            return self.ecod_tfgroup_dict[pdb_code]
        except Exception as e:
            # print(f"Failing to get T/F group for {pdb_code}: {e}")
            return []

    def load_ecod_dict(self):
        save_path = Path(self.tmp_dir, self.ecod_url.split('/')[-1])
        remote_dl(self.ecod_url, str(save_path), uncompress=True)

        ecod_tfgroup_dict = {}
        with open(save_path, "r") as f:
            lines = f.readlines()[6:]  # start by line 6
            """
            #/data/ecod/database_versions/v280/ecod.develop280.domains.txt
            #ECOD version develop280
            #Domain list version 1.6
            #Grishin lab (http://prodata.swmed.edu/ecod)
            #uid	ecod_domain_id	manual_rep	f_id	pdb	chain	pdb_range	seqid_range	unp_acc	arch_name	x_name	h_name	t_name	f_name
            002728551	e7d2xA1	AUTO_NONREP	1.1.1	7d2x	A	A:-3-183	A:20-206	NO_UNP	beta barrels	"cradle loop barrel"	"RIFT-related"	"acid protease"	F_UNCLASSIFIED
            002728572	e7d5aA2	AUTO_NONREP	1.1.1	7d5a	A	A:-3-183	A:20-206	NO_UNP	beta barrels	"cradle loop barrel"	"RIFT-related"	"acid protease"	F_UNCLASSIFIED
            002726563	e7b1eA1	AUTO_NONREP	1.1.1	7b1e	A	A:46P-183	A:14-199	NO_UNP	beta barrels	"cradle loop barrel"	"RIFT-related"	"acid protease"	F_UNCLASSIFIED
            002726573	e7b1pA2	AUTO_NONREP	1.1.1	7b1p	A	A:47P-183	A:15-199	NO_UNP	beta barrels	"cradle loop barrel"	"RIFT-related"	"acid protease"	F_UNCLASSIFIED
            """

            print("Length of input ECOD name list", len(lines))
            for line in lines:
                ll = line.split("\t")
                entryId = ll[4].lower()
                tGroup = ll[12].replace('"', "")
                fGroup = ll[13].replace('"', "")
                Group = tGroup + "|" + fGroup

                if entryId not in ecod_tfgroup_dict:
                    ecod_tfgroup_dict[entryId] = [Group]
                else:
                    ecod_tfgroup_dict[entryId].append(Group)
        return ecod_tfgroup_dict

    def get_skempi_code(self):
        save_path = Path(self.tmp_dir, self.skempi_url.split('/')[-1])
        remote_dl(self.skempi_url, str(save_path), uncompress=False)

        code_list = []
        with open(save_path, "r") as f:
            """
            #Pdb;Mutation(s)_PDB;Mutation(s)_cleaned;iMutation_Location(s)
            1CSE_E_I;LI45G;LI38G;COR;Pr/PI;Pr/PI;5.26E-11;5.26E-11;1.12E-12
            """
            lines = f.readlines()[1:]  # start by line 1
            for line in lines:
                line = line.split("_")
                code = str(line[0]).lower()
                code_list.append(code)

        return list(set(code_list))

    def get_3dcomplex_code(self):
        save_path = Path(self.tmp_dir, self.complex_QS40_url.split('/')[-1])
        remote_dl(self.complex_QS40_url, str(save_path), uncompress=False)

        code_list = []
        with open(save_path, "r") as f:
            """
            # 5f87_6
            # 5d06_1
            """
            lines = f.readlines()
            for line in lines:
                line = line.split("_")
                code = str(line[0]).lower()
                code_list.append(code)

        return list(set(code_list))

    def get_pdbbind_code(self):
        save_path = Path(self.tmp_dir, self.pdbbind_index_url.split('/')[-1])
        remote_dl(self.pdbbind_index_url, str(save_path), uncompress=True)

        file_name = Path(save_path, "index", self.PP_complex_file)
        code_list = []
        with open(file_name, "r") as f:
            lines = f.readlines()[6:]  # start by line 7
            """
            3sgb  1.80  1983  Kd=17.9pM     // 3sgb.pdf (56-mer) TURKEY OVOMUCOID INHIBITOR (OMTKY3), 1.79 x 10-11M
            2tgp  1.90  1983  Kd~2.4uM      // 2tgp.pdf (58-mer) TRYPSIN INHIBITOR, 2.4 x 10-6M
            1ihs  2.00  1994  Ki=0.3nM      // 1ihs.pdf (21-mer) hirutonin-2 with human a-thrombin, led to Ki=0.3nM
            """
            for line in lines:
                line = line.split()
                code = str(line[0]).lower()
                code_list.append(code)

        return list(set(code_list))


def pdb_download(pdb_code, pdb_dir, pdb_source):
    if pdb_source not in PDB_DOWNLOAD_SOURCES:
        raise ValueError(
            f"Unknown {pdb_source=}, must be one of {tuple(PDB_DOWNLOAD_SOURCES)}"
        )
    download_url_template = PDB_DOWNLOAD_SOURCES[pdb_source]
    filename = os.path.join(pdb_dir, f"{pdb_code}_{pdb_source}.pdb".lower())
    download_url = download_url_template.format(pdb_code=pdb_code).lower()

    uncompress = download_url.endswith(("gz", "gzip", "zip"))
    return remote_dl(download_url, filename, uncompress=uncompress, skip_existing=True, verbose=False)


def parallel_download(final_codes, pdb_dir, pdb_source, max_workers=10):
    with tqdm(total=len(final_codes), desc='Downloading PDBs') as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for code in final_codes:
                futures.append(executor.submit(pdb_download, code, pdb_dir, pdb_source))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")
                pbar.update(1)
                
            
def remote_dl(url, save_path, uncompress=False, skip_existing=False, verbose=True):
    if skip_existing:
        if os.path.isfile(save_path) and verbose: #os.path.getsize(save_path) > 0:
            print(f"File {save_path} exists, skipping download...")
            return save_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        if 300 <= r.status_code < 400:
            raise HTTPError(f"Redirect {r.status_code} for url{url}", response=r)

        save_dir = Path().joinpath(*Path(save_path).parts[:-1])
        os.makedirs(save_dir, exist_ok=True)

        if uncompress:
            file_like_object = io.BytesIO(r.content)
            if url.endswith('.tar.gz'):
                with tarfile.open(fileobj=file_like_object, mode='r:gz') as tar:
                    tar.extractall(save_path)
            elif url.endswith((".gz", ".gzip")):
                with gzip.open(file_like_object, 'rb') as f_in:
                    with open(save_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            elif url.endswith('.zip'):
                with zipfile.ZipFile(file_like_object, 'r') as zip_ref:
                    zip_ref.extractall(save_path)
            else:
                print(f"Unknown compress format for url{url}")

        else:
            with open(save_path, 'wb') as out_handle:
                try:
                    in_handle = r.raw
                    out_handle.write(in_handle.read())

                finally:
                    in_handle.close()
        if verbose:
            print(f"Downloaded {save_path} from {url}")
        return save_path


def cleanup_temp_dir(tmp_dir):
    """
    Deletes the temporary directory and its contents.
    """
    try:
        shutil.rmtree(tmp_dir)
        print(f"Deleted temporary directory: {tmp_dir}")
    except Exception as e:
        print(f"Error occurred while deleting temporary directory {tmp_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--pdb_source', type=str, default='rc')
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--clean_up', action='store_true', default=True)
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temporary directory: {temp_dir}")

    Fetcher = PdbCodeFetcher(temp_dir)
    final_codes = Fetcher.get_final_code()

    pdb_dir = os.path.abspath(args.pdb_dir)
    os.makedirs(pdb_dir, exist_ok=True)

    print(f"Created pdb directory: {pdb_dir}")
    parallel_download(final_codes, pdb_dir, args.pdb_source, args.max_workers)
    
    print("Done with downloading PDBs!")

    if args.clean_up:
        cleanup_temp_dir(temp_dir)

