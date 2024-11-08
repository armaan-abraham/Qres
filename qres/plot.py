import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from Bio.PDB import PDBParser
import torch

# Instantiate the StructurePredictor
from qres.structure_prediction import StructurePredictor

def plot_structure(sequence: str, output_path: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = StructurePredictor(device=device)

    # Predict the structure
    pdb_str = predictor.predict_structure([sequence])[0]

    # Parse PDB data
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_str))

    # Extract atom coordinates
    atoms = []
    for atom in structure.get_atoms():
        atoms.append(atom.get_coord())
    atoms = np.array(atoms)

    # Plot the structure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], s=20, c="b", alpha=0.6)

    # Set plot labels and title
    ax.set_title("Predicted 3D Structure")
    ax.set_xlabel("X-axis (Å)")
    ax.set_ylabel("Y-axis (Å)")
    ax.set_zlabel("Z-axis (Å)")

    # Save the figure
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    sequences = [
        "",
        ""
    ]
    """
      Step 1   : Sequence: AQVWNACCHTKAQDWCCCMRIDEGKRSCYG, Reward: 0.46955960988998413
        Step 2   : Sequence: AQVWNACCHTKAQLWCCCMRIDEGKRSCYG, Reward: 0.46335723996162415
        Step 3   : Sequence: AQVWNACCHTKAQLWCCCMLIDEGKRSCYG, Reward: 0.4842478632926941
        Step 4   : Sequence: AQVWNACCHTKAQLWCLCMLIDEGKRSCYG, Reward: 0.787070095539093
        Step 5   : Sequence: AQVWNACCHTLAQLWCLCMLIDEGKRSCYG, Reward: 0.8204230070114136
        Step 6   : Sequence: AQVWNACCHTLLQLWCLCMLIDEGKRSCYG, Reward: 0.8204230070114136
        """
