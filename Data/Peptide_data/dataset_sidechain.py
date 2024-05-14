from typing import List, Tuple, Union, Optional, Dict
import os
import logging
from math import isinf, floor, ceil, log
import tarfile
from uuid import uuid4
from tempfile import gettempdir

import h5py
import numpy
import torch
from glob import glob
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from Bio.PDB.Polypeptide import is_aa, one_to_three

from openfold.openfold.np.residue_constants import restype_atom37_mask, restype_atom14_mask, chi_angles_mask, restypes
from openfold.openfold.data.data_transforms import (atom37_to_frames,
                                           atom37_to_torsion_angles,
                                           get_backbone_frames,
                                           make_atom14_masks,
                                           make_atom14_positions)
from openfold.openfold.utils.feats import atom14_to_atom37

def _read_residue_data(residues: List[Residue]) -> Dict[str, torch.Tensor]:
    """
    Convert residues from a structure into a format that SwiftMHC can work with.
    (these are mostly openfold formats, created by openfold code)

    Args:
        residues: from the structure

    Returns:
        residue_numbers: [len] numbers of the residue as in the structure
        aatype: [len] sequence, indices of amino acids
        sequence_onehot: [len, 22] sequence, one-hot encoded amino acids
        blosum62: [len, 20] sequence, BLOSUM62 encoded amino acids
        backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
        torsion_angles_sin_cos: [len, 7, 2] representations of the torsion angles (one sin & cos per angle)
        alt_torsion_angles_sin_cos: [len, 7, 2] representations of the alternative torsion angles (one sin & cos per angle)
        torsion_angles_mask: [len, 7] which torsion angles each residue has (openfold format)
        atom14_gt_exists: [len, 14] which atoms each residue has (openfold 14 format)
        atom14_gt_positions: [len, 14, 3] atom positions (openfold 14 format)
        atom14_alt_gt_positions: [len, 14, 3] alternative atom positions (openfold 14 format)
        residx_atom14_to_atom37: [len, 14] per residue, conversion table from openfold 14 to openfold 37 atom format
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # embed the sequence
    amino_acids = [amino_acids_by_code[r.get_resname()] for r in residues]
    sequence_onehot = torch.stack([aa.one_hot_code for aa in amino_acids]).to(device=device)
    aatype = torch.tensor([aa.index for aa in amino_acids], device=device)

    # get atom positions and mask
    atom14_positions = []
    atom14_mask = []
    residue_numbers = []
    for residue_index, residue in enumerate(residues):
        p, m = get_atom14_positions(residue)
        atom14_positions.append(p.float())
        atom14_mask.append(m)
        residue_numbers.append(residue.get_id()[1])

    atom14_positions = torch.stack(atom14_positions).to(device=device)
    atom14_mask = torch.stack(atom14_mask).to(device=device)
    residue_numbers = torch.tensor(residue_numbers, device=device)

    # convert to atom 37 format, for the frames and torsion angles
    protein = {
        "residue_numbers": residue_numbers,
        "aatype": aatype,
        "sequence_onehot": sequence_onehot,
        "blosum62": blosum62,
    }

    protein = make_atom14_masks(protein)

    atom37_positions = atom14_to_atom37(atom14_positions, protein)
    atom37_mask = atom14_to_atom37(atom14_mask.unsqueeze(-1), protein)[..., 0]

    protein["atom14_atom_exists"] = atom14_mask
    protein["atom37_atom_exists"] = atom37_mask

    protein["all_atom_mask"] = atom37_mask
    protein["all_atom_positions"] = atom37_positions

    # get frames, torsion angles and alternative positions
    protein = atom37_to_frames(protein)
    protein = atom37_to_torsion_angles("")(protein)
    protein = get_backbone_frames(protein)
    protein = make_atom14_positions(protein)

    return protein

def get_atom14_positions(residue: Residue) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the positions of the atoms for one residue.
    Args:
        residue: the residue to get the atoms for
    Returns:
        the residue's atom positions in openfold atom14 format
        the masks for the atoms in openfold atom14 format
    """
    atom_names = openfold_residue_atom14_names[residue.get_resname()]
    masks = []
    positions = []
    for atom_name in atom_names:
        if len(atom_name) > 0:
            try:
                atom = _get_atom(residue, atom_name)
                positions.append(atom.coord)
                masks.append(True)
            except Exception as e:
                masks.append(False)
                positions.append((0.0, 0.0, 0.0))
                _log.warning(f"{residue.get_full_id()} not adding 14-formatted position for atom: {str(e)}")
        else:
            masks.append(False)
            positions.append((0.0, 0.0, 0.0))
    return torch.tensor(numpy.array(positions)), torch.tensor(masks)

if __name__ == '__main__':

    pdb_parser = PDBParser()
    pdb_dir = '/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/'
    out_file_name = './Data/Peptide_data/sidechain_100K.hdf5'
    for file_path in glob('/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pdb/*.pdb'):

        structure_object = pdb_parser.get_structure(id, file_path)
        structure_model = structure_object.get_models()[0]
        for chain in structure_model.get_chains():

            residues = list(chain.get_residues())

            residue_data = _read_residue_data(residues)

            with h5py.File(out_file_name, 'a') as f:
                main_group = f.require_group(os.path.basename(file_path))
                chain_group = main_group.require_group(chain.id)
                for key, value in residue_data.items():
                    chain_group.create_dataset(key, data=value.numpy())