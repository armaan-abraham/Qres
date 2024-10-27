import io
from typing import List

import torch
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

torch.backends.cuda.matmul.allow_tf32 = True


# Ensure that only one model is loaded in memory
class StructurePredictor:
    def __init__(self, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
        ).to(device)
        self.model.esm = self.model.esm.half()

    def predict_structure(self, sequences: List[str]) -> List[str]:
        tokenized_input = self.tokenizer(
            sequences, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)["input_ids"]
        with torch.no_grad():
            outputs = self.model(tokenized_input)
        outputs = self.convert_outputs_to_pdb(outputs)
        return outputs

    def convert_outputs_to_pdb(self, outputs):
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
                chain_index=outputs["chain_index"][i]
                if "chain_index" in outputs
                else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def overall_confidence_from_pdb(self, pdb):
        """
        Parses PDB format data from a string and calculates the overall confidence of the structure
        based on the average B-factor of the atoms.
        """
        # Create a StringIO object from the string data
        pdb_io = io.StringIO(pdb)

        # Initialize the PDB parser
        parser = PDBParser(QUIET=True)

        try:
            # Use the parser to create a structure object from the StringIO object
            structure = parser.get_structure("PDB_structure", pdb_io)
        except Exception:
            # Return zero confidence if parsing fails
            return 0.0

        # Collect B-factors
        b_factors = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        b_factors.append(atom.get_bfactor())

        # Compute the average B-factor
        if b_factors:
            average_b_factor = sum(b_factors) / len(b_factors)
            return average_b_factor
        else:
            return 0.0  # No B-factors found
