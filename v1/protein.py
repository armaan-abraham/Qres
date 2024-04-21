import io
from collections import namedtuple
from pathlib import Path
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

import numpy as np
from Bio.PDB import PDBParser

from fold import infer_structure_batch

PDB_CACHE_PATH = Path(__file__.parent / "pdb")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_LENGTH_ANGSTROM = 3.8

torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

assert torch.cuda.is_available(), "You must be dumb to try to run this on a CPU"
model = model.cuda()

model.esm = model.esm.half()


def infer_structure_batch(sequences):
    tokenized_input = tokenizer(
        sequences, return_tensors="pt", add_special_tokens=False
    ).to(model.device)["input_ids"]
    if torch.cuda.is_available():
        tokenized_input = tokenized_input.cuda()
    with torch.no_grad():
        outputs = model(tokenized_input)
    start = time.time()
    outputs = convert_outputs_to_pdb(outputs)
    print("convert outputs to pdb", time.time() - start)
    return outputs


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def get_max_physical_protein_length_A(n_amino_acids):
    return (n_amino_acids - 1) * AMINO_ACID_LENGTH_ANGSTROM


def make_pdb_path(sequence):
    # compute hash of sequence
    hashed_sequence = hash(sequence)
    return PDB_CACHE_PATH / f"P{hashed_sequence}.pdb"


def generate_pdbs(sequences):
    pdbs = infer_structure_batch(sequences)
    for sequence, pdb in zip(sequences, pdbs):
        pdb_path = make_pdb_path(sequence)
        with open(pdb_path, "w") as f:
            f.write(pdb)
    return pdbs


def load_pdbs(sequences):
    # load pdbs from pdb cache
    pdbs = []
    for sequence in sequences:
        pdb_path = make_pdb_path(sequence)
        if pdb_path.exists():
            with open(pdb_path, "r") as f:
                pdbs.append(f.read())
        else:
            raise FileNotFoundError(f"Could not find PDB file for sequence {sequence}")


# TODO check that this outputs angstroms
def get_distance_matrix(pdb):
    """
    Generates a distance matrix for the Cα atoms of each residue in a PDB file.

    Args:
    pdb (str): PDB file contents.

    Returns:
    numpy.ndarray: A 2D array where element (i, j) is the distance between the Cα of residue i and the Cα of residue j.
    """
    # Create a StringIO object from the string data
    pdb_io = io.StringIO(pdb)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_io)
    pdb_io.close()

    ca_atoms = []

    # Collect all Cα atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue:
                    ca_atoms.append(residue["CA"])

    # Number of residues
    num_residues = len(ca_atoms)
    distance_matrix = np.zeros((num_residues, num_residues))

    # Compute distances
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            distance = ca_atoms[i].coord - ca_atoms[j].coord
            distance_matrix[i][j] = distance_matrix[j][i] = np.sqrt(
                np.sum(distance * distance)
            )

    return distance_matrix


def flatten_distance_matrix(distance_matrix):
    """
    Flattens a distance matrix into a 1D array.

    Args:
    distance_matrix (numpy.ndarray): A 2D array representing a distance matrix.

    Returns:
    numpy.ndarray: A 1D array containing the upper triangular part of the distance matrix.
    """
    result = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    assert result.size == distance_matrix.shape[0] ** 2 / 2 - distance_matrix.shape[0]
    return result


def get_amino_acids_from_pdb(pdb):
    """
    Reads a PDB file and returns a list of amino acids.

    Args:
    pdb (str): PDB file contents.

    Returns:
    list of tuples: Each tuple contains (chain ID, residue name, residue sequence number).
    """
    pdb_io = io.StringIO(pdb)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_io)
    pdb_io.close()
    amino_acids = []  # List to hold the amino acids

    for model in structure:  # Usually, there's only one model in a PDB file
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":  # Filter out heteroatoms and water
                    amino_acids.append((chain.id, residue.resname, residue.id[1]))

    return amino_acids


def compute_quaternions(pdb):
    """
    Computes quaternions between consecutive Cα atoms in a protein structure.

    Args:
    pdb (str): PDB file contents.

    Returns:
        list of np.quaternion: List of quaternions between consecutive Cα atoms.
    """
    pdb_io = io.StringIO(pdb)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_io)
    pdb_io.close()
    ca_atoms = []

    # Collect Cα atoms
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"])

    # Calculate vectors between consecutive Cα atoms
    vectors = []
    for i in range(len(ca_atoms) - 1):
        vec = ca_atoms[i + 1].coord - ca_atoms[i].coord
        vectors.append(vec / np.linalg.norm(vec))  # Normalize the vector

    # Compute quaternions
    quaternions = []
    for i in range(len(vectors) - 1):
        q = quaternion_from_vectors(vectors[i], vectors[i + 1])
        quaternions.append(q)

    return quaternions


def flatten_quaternions(quaternions):
    return np.array(quaternions).flatten()


def quaternion_from_vectors(v1, v2):
    """
    Computes the quaternion required to rotate vector v1 to vector v2.

    Args:
        v1, v2 (np.array): Vectors between which the quaternion is computed.

    Returns:
        np.quaternion: Quaternion representing the rotation from v1 to v2.
    """
    # Calculate the cross and dot products
    v_cross = np.cross(v1, v2)
    v_dot = np.dot(v1, v2)

    # Compute quaternion components
    q = np.zeros(4)
    q[0] = np.sqrt(np.sum(v1**2) * np.sum(v2**2)) + v_dot  # scalar part
    q[1:] = v_cross  # vector part

    # Normalize the quaternion
    q /= np.linalg.norm(q)
    return q


if __name__ == "__main__":
    test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

    with open("test_protein.txt", "w") as f:
        # write the file
        result = infer_structure_batch([test_protein[:50]])[0]
        f.write(result)

    exit()
    tpp = []
    for i in range(15):
        test_proteins = [test_protein[:50] for j in range(i + 1)]
        start = time.time()
        infer_structure_batch(test_proteins)
        tpp.append((time.time() - start) / (i + 1))
    ax = sns.lineplot(x=range(len(tpp)), y=tpp)
    # print to svg
    plt.savefig("test.svg")
