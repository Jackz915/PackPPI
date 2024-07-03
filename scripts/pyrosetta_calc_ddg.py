import tempfile
import os
from Bio.PDB import PDBParser, PDBIO, Select

import pyrosetta
from pyrosetta import rosetta
from pyrosetta import init
from pyrosetta.rosetta import *
from pyrosetta.toolbox import *
from pyrosetta.teaching import *

def mut_init():
    init('-mute all '
         '-ignore_unrecognized_res 1 '
         '-ex1 '
         '-ex2 '
         '-flip_HNQ '
         '-relax:cartesian '
         '-nstruct 200 '
         '-crystal_refine ' 
         '-optimization:default_max_cycles 200 ')

def parse_mutation(mutstr):
    """
    Parse a mutation string to extract wild-type residue, chain ID, position, and mutant residue.
    
    Parameters:
    mutstr (str): A string representing a single mutation (e.g., "RA47A").
    
    Returns:
    tuple: A tuple containing (wild-type residue, chain ID, position, mutant residue).
    
    Raises:
    ValueError: If the mutation string is invalid or parsing fails.
    """
    if len(mutstr) < 4:
        raise ValueError(f"Invalid mutation string format: {mutstr}")
    
    wt_residue = mutstr[0]
    mutant_residue = mutstr[-1]
    
    # Initialize variables
    chain_id = ''
    position = None
    
    # Find the position where the numeric part starts
    for i in range(1, len(mutstr)):
        if mutstr[i].isdigit():
            if i > 1 and mutstr[i-1].isalpha():
                chain_id = mutstr[1:i]
            position = int(mutstr[i:-1])
            break
    
    # Raise an error if parsing failed
    if position is None:
        raise ValueError(f"Failed to parse mutation string: {mutstr}")
    
    return wt_residue, chain_id, position, mutant_residue


def parse_mutations(mutations_str):
    """
    Parse a string containing one or more mutations separated by commas or a single mutation string.
    
    Parameters:
    mutations_str (str): A string containing one or more mutations separated by commas (e.g., "RA47A,EA48A,HA52A" or "RA47A").
    
    Returns:
    list: A list of tuples, each containing (wild-type residue, chain ID, position, mutant residue).
    
    Raises:
    ValueError: If the mutation string is invalid or parsing fails.
    """
    mutation_list = mutations_str.split(',')
    parsed_mutations = [parse_mutation(mutstr) for mutstr in mutation_list]
    return parsed_mutations


def validate_residue(residue):
    """Validate if the residue type is in the standard amino acid set."""
    standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    if residue not in standard_amino_acids:
        raise ValueError(f"Invalid residue type: {residue}. Must be one of {standard_amino_acids}.")


def clean_pdb_with_cleanATOM(pdb_path):
    # Use PyRosetta's prebuilt cleaner function
    cleanATOM(pdb_path)


def calculate_ddg(pdb_path, mutations_str):

    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxnDDG = get_fa_scorefxn()

    # Energy minimization  setup
    min_mover = rosetta.protocols.minimization_packing.MinMover()  # define a Mover in type of MinMover

    #Initializations for Movers
    mm = MoveMap()
    mm.set_bb_true_range(28, 36)
    min_mover.movemap(mm)
    #Score with our score we defined
    min_mover.score_function(scorefxnDDG)
    
    #Literature sugges .01 here seems to not make huge difference in output of MC
    min_mover.tolerance(0.01)
    
    def minimize_Energy(pose):
        # Minimization
        min_mover.apply(pose)
    
        # Trial_mover define
        #Kt normal at 1 adjust if you want a different kT This is usually higher at high heats ~1 = ~ physiological 
        kT = 1.0
    
        # make mc a monteCarlo with our scoring and our Kt
        mc = MonteCarlo(pose, scorefxnDDG, kT)
        mc.boltzmann(pose)
        mc.recover_low(pose)
    
        #setup needed for MC
        trial_mover = TrialMover(min_mover, mc)
        #Run Monte Carlo
        #iteartions set to 100 as of now change if results necessitate 
        for i in range(100):
            trial_mover.apply(pose)
        
    mutant_pose = Pose()
    mutations = parse_mutations(mutations_str)
        
    for wt_residue, chain_id, position, mutant_residue in mutations:
        mutant_pose.assign(pose)
        validate_residue(wt_residue)
        validate_residue(mutant_residue)
        print(f"Applying mutation: {wt_residue}{chain_id}{position}{mutant_residue}")
        
        # Convert chain ID to pose chain index
        try:
            pose_residue_index = mutant_pose.pdb_info().pdb2pose(chain_id, position)
        except RuntimeError:
            raise ValueError(f"Failed to convert chain ID '{chain_id}' and position '{position}' to pose index.")
        
        if pose_residue_index == 0:
            print(f"Invalid residue position {position} in chain {chain_id} for pose.")
            continue
            #raise ValueError(f"Invalid residue position {position} in chain {chain_id} for pose.")
        
        # Apply mutation using mutate_residue
        mutate_residue(mutant_pose, pose_residue_index, mutant_residue)
        minimize_Energy(mutant_pose)
        
    # Calculate ddG
    wildtype_score = scorefxnDDG(pose)
    mutant_score = scorefxnDDG(mutant_pose)
    ddg = mutant_score - wildtype_score
    
    return ddg
