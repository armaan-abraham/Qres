import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
from Bio.PDB import PDBParser

from qres.fold import infer_structure_batch

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_LENGTH_ANGSTROM = 3.8


def flattened_quaternions_length(n_amino_acids):
    return (n_amino_acids - 2) * 4


def flattened_distance_matrix_length(n_amino_acids):
    return (n_amino_acids * (n_amino_acids + 1)) / 2 - n_amino_acids


def get_max_physical_protein_length_A(n_amino_acids):
    return (n_amino_acids - 1) * AMINO_ACID_LENGTH_ANGSTROM


def generate_pdbs(sequences):
    pdbs = infer_structure_batch(sequences)
    return pdbs


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
    assert result.size == flattened_distance_matrix_length(distance_matrix.shape[0])
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
