import os
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple
from src.utils.protein import to_pdb
import freesasa
from Bio.PDB import PDBParser as BioParser, Selection, NeighborSearch


def get_interface_residues(pdb_file, radius=5.0):
    """Return a list of interacting residues based on accessibility and distance to partner

    :param str pdb_file: PDB file path
    :param int,float radius: Maximum distance to be considered to define a residue as interacting with another chain
    :return: interface: Dictionary of interface residues per chain IDs (e.g. {'A':[1,2,3], 'B': [10,11,12], ...}
    :rtype: dict

    """
    p = BioParser(QUIET=True)
    s = p.get_structure('pdb', pdb_file)
    if sum(1 for _ in s.get_chains()) < 2:
        return None
    m = s[0]
    all_atoms = Selection.unfold_entities(m, 'A')
    # Unfold atoms for NeighborSearch algorithm to work
    ns = NeighborSearch(all_atoms)
    interface = ns.search_all(radius, "R")
    # Filter redundant residues
    buffer = dict([(ch_id.id, []) for ch_id in m.get_chains()])
    for r in interface:
        if r[0].parent.id != r[1].parent.id:
            if r[0].id[1] not in buffer[r[0].parent.id]:
                buffer[r[0].parent.id].append(r[0].id[1])
            if r[1].id[1] not in buffer[r[1].parent.id]:
                buffer[r[1].parent.id].append(r[1].id[1])
    interface = buffer
    for ch_id in m.get_chains():
        interface[ch_id.id] = sorted(interface[ch_id.id])
    return interface


def extract_interface(protein: Dict, chain_main: str, outdir: Path):
    """Generate a set of dimers from a given protein, a chain of interest that will be present in all dimers
        and the other chains to be used.

        For instance, if:
            - protein "xxxx.pdb" has 4 chains: A, B, C and D
            - chain_main is defined as the chain B

        then the script will generate PDB files of the following dimers:
            - AB, BC and BD

    Args:
        protein (Dict): A dictionary containing protein information
        chain_main (str): The main chain to be present in all dimers.
        outdir (Path): The directory to store output files.
    """

    def write_pdb(filename: Path, protein: Dict, chains: List[str]) -> None:
        """Writes the specified chains of the protein to a PDB file."""
        pdb_content = to_pdb(prot=protein, keep_chains=chains)
        with open(filename, 'w') as file:
            file.writelines(pdb_content)

    def compute_sasa(pdb_file: Path) -> Dict:
        """Computes the solvent accessible surface area for the given PDB file."""
        return getResiduesArea(pdbfile=pdb_file)

    os.makedirs(outdir, exist_ok=True)
    
    # Get chains in structure that are not the main chain
    chain_others = [chain for chain in np.unique(protein["chain_id"]) if chain != chain_main]

    # Write the PDB file for the given chain
    pdb_chain_filename = Path(outdir, f"tem_{chain_main}.pdb")
    write_pdb(pdb_chain_filename, protein, [chain_main])

    # List of all possible dimers that can be formed with the main chain
    dimer_combinations = [x for x in list(combinations([chain_main] + chain_others, 2)) if chain_main in ''.join(x)]
    
    # Generate dimers
    dimer_filenames = []
    for dimer_chains in dimer_combinations:
        dimer_filename = Path(outdir, f"tem_{''.join(dimer_chains)}.pdb")
        dimer_filenames.append(dimer_filename)
        write_pdb(dimer_filename, protein, [dimer_chains])

    # Compute SASA for the studied chain
    residues_area_chain = compute_sasa(pdb_chain_filename)

    # Compute SASA for each dimer and determine interface residues
    residues_interfaces_chain = []
    for dimer_file in dimer_filenames:
        residues_area_dimer = compute_sasa(dimer_file)
        _, interface_chain_dimer = getSurfRes(structure_complex=residues_area_dimer,
                                              structure_receptor=residues_area_chain)
        if interface_chain_dimer:
            residues_interfaces_chain.append(interface_chain_dimer)

    # Write file listing interface residues of the given chain
    interface_filename = Path(outdir, f"tem_{chain_main}_interface.txt")
    write_interface_residues(interface_map=residues_interfaces_chain, outfile=interface_filename)

    return dimer_combinations



def getSurfRes(structure_complex: dict, structure_receptor: dict) -> Tuple[list, list]:
    """Extract surface and interface residues through the comparison of Accessible Surface Area (ASA)
    of a freesasa structure complex and a freesasa structure of a part of the complex (called receptor).

    For instance, let's define A_resx_B a complex AB where A interacts with B through the residue resx.
    - structure_complex -> freesasa structure of AB
    - structure_receptor -> freesasa structure of A (or B)

    If ASA of a resn in A is higher than ASA of resn in AB, then resn is part of the interface between A and B.

    Example of freesasa attributes of a residue:
    >>> structure_receptor['A']['254']
    >>> {
        'residueType': 'PHE',
        'residueNumber': '254',
        'total': 17.181261911668134,
        'mainChain': 5.838150815071199,
        'sideChain': 11.343111096596939,
        'polar': 2.0559216586027156,
        'apolar': 15.12534025306542,
        'hasRelativeAreas': True,
        'relativeTotal': 0.08595788428891402,
        'relativeMainChain': 0.15191649271587818,
        'relativeSideChain': 0.07025773364259486,
        'relativePolar': 0.05884148994283674,
        'relativeApolar': 0.09170207501555366
    }

    Args:
        structure_complex (dict): freesasa Structure
        structure_receptor (dict): freesasa Structure

    Returns:
        tuple[list, list]:
            - surface_receptor (list): surface residues of the receptor. Each element of the list is a freesasa structure residue with an additional 'chain" attribute.
            - interface_receptor (list): interface residues of receptor. Each element of the list is a freesasa structure residue with an additional 'chain" attribute.
    """

    # init var
    surface_receptor = []
    interface_receptor = []

    for chain_receptor in list(structure_receptor.keys()):

        # loop over cplx chains
        for chain_complex in structure_complex:
            if chain_complex != chain_receptor:
                continue

            # loop over residues
            for residue_complex in structure_complex[chain_complex]:
                freesasa_residue_receptor = structure_receptor[chain_complex][str(residue_complex)]
                freesasa_residue_complex = structure_complex[chain_complex][str(residue_complex)]

                delta_ASA = freesasa_residue_receptor.relativeTotal - freesasa_residue_complex.relativeTotal
                freesasa_residue_receptor.chain = chain_receptor

                if delta_ASA > 0:  # means the ASA in the lig is greater than in the cplx form, so the res is buried in the interface
                    interface_receptor.append(freesasa_residue_receptor)
                    surface_receptor.append(freesasa_residue_receptor)
                    # print(f"- residue {freesasa_residue_receptor.residueType}-{freesasa_residue_receptor.residueNumber} of chain {chain_receptor} is part of interface")
                else:  # not in interface
                    if structure_receptor[chain_complex][str(residue_complex)].relativeTotal > 0.25:  # means the res is located on the lig surface
                        surface_receptor.append(freesasa_residue_receptor)

    return surface_receptor, interface_receptor


def write_interface_residues(interface_map: list, outfile: Path) -> None:
    """
    Write a text file listing interface residues of a chain.

    The file is formatted with the following informations (space or tab separated):
    "chain" "resid" "resname" "label value"

    Example:
    #chain   #resid   #resname #label_value
    A   2    PHE    1
    A   3    GLU    1
    A   6    ALA    1

    """

    with open(outfile, "w") as _f:
        for i, interface in enumerate(interface_map, start=1):
            for residue in interface:
                _f.write(f"{residue.chain}\t{residue.residueNumber}\t{residue.residueType}\t{i}\n")


def getResiduesArea(pdbfile: Path) -> dict:
    """Get SASA for all residues including relative areas if available for the classifier used.

    Returns
    Relative areas are normalized to 1, but can be > 1 for residues in unusual conformations or at the ends of chains.

    Args:
        pdbfile (str): Absolut or relative path of a PDB filename

    Returns:
        dict:   dictionary of results where first dimension is chain label and the second dimension residue number:
                >>>result["A"]["5"]
                -> gives the ResidueArea of residue number 5 in chain A.
    """
    pdbfile = str(pdbfile)
    freesasa.setVerbosity(1)
    structure = freesasa.Structure(pdbfile)
    sasa = freesasa.calc(structure)

    return sasa.residueAreas()


def parse_interface_file(interface_dir: Path) -> dict:

    inter_residues = {}
    for file_name in os.listdir(interface_dir):
        if 'interface' in file_name:
            file_path = Path(interface_dir, file_name)
            with open(file_path, "r") as _file:
                for line in _file:
                    if line.startswith("#") or not line:
                        continue

                    chain, resid, resname, site_label = line.split()
                    if chain not in inter_residues:
                        inter_residues[chain] = []
                    inter_residues[chain].append(resid)

            os.remove(file_path)

    return inter_residues





