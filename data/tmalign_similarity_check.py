import os
import subprocess
import argparse
import concurrent.futures
from tqdm import tqdm

def run_tmalign(pdb1, pdb2, tmalign_path):
    command = f"{tmalign_path} {pdb1} {pdb2}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def extract_similarity(output):
    tm_scores = []
    for line in output.splitlines():
        if "TM-score" in line:
            parts = line.split()
            for part in parts:
                try:
                    score = float(part)
                    tm_scores.append(score)
                except ValueError:
                    continue 
    if tm_scores:
        return sum(tm_scores) / len(tm_scores)  
    return 0.0

def check_similarity_and_remove(pdb_dir1, pdb_dir2, threshold=0.5, tmalign_path="TMalign", max_workers=10):
    pdb_files1 = [f for f in os.listdir(pdb_dir1) if f.endswith('.pdb')]
    pdb_files2 = [f for f in os.listdir(pdb_dir2) if f.endswith('.pdb')]
    removed_count = 0 

    def process_file(pdb1):
        nonlocal removed_count 
        for pdb2 in pdb_files2:
            output = run_tmalign(os.path.join(pdb_dir1, pdb1), os.path.join(pdb_dir2, pdb2), tmalign_path)
            similarity = extract_similarity(output)
            if similarity > threshold:
                os.remove(os.path.join(pdb_dir1, pdb1))
                removed_count += 1 
                print(f"Removed {pdb1} due to similarity with {pdb2}")
                return  

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_file, pdb_files1), total=len(pdb_files1), desc="Processing files"))

    print(f"Total removed files: {removed_count}") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str, required=True, help='Path to the first PDB directory')
    parser.add_argument('--dir2', type=str, required=True, help='Path to the second PDB directory')
    parser.add_argument('--tmalign_path', type=str, required=True, help='Path to TMalign executable')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold for removal')
    parser.add_argument('--max_workers', type=int, default=80, help='Number of threads to use')
    args = parser.parse_args()

    check_similarity_and_remove(args.dir1, args.dir2, args.threshold, args.tmalign_path, args.max_workers)

if __name__ == "__main__":
    main()
