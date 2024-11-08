import io
import torch
import py3Dmol
from qres.structure_prediction import StructurePredictor

def plot_structure(sequence: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = StructurePredictor(device=device)

    # Predict the structure
    pdb_str = predictor.predict_structure([sequence])[0]

    # Visualize using py3Dmol
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_str, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()

    # Render the viewer and save as an image
    png = view.png()

    # Save the PNG image to the output path
    with open(output_path, 'wb') as f:
        f.write(png)


if __name__ == "__main__":
    sequences = ["IWVRNSRNWWWWVTMNDWNN", "IWVRNSRNWWWWVTMNLWNN"]
    plot_structure(sequences[0], "plot_1.png")
