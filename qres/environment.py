import torch
import torch.nn.functional as F
from typing import Tuple, List

from qres.env import infer_structure_batch, overall_confidence_from_pdb

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

class ProteinEnv:
    def __init__(self, config):
        self.config = config
        
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """One-hot encode an amino acid sequence."""
        idx = torch.tensor([AMINO_ACIDS.index(aa) for aa in sequence], dtype=torch.long)
        return F.one_hot(idx, num_classes=len(AMINO_ACIDS)).float().flatten()
    
    def decode_sequence(self, encoded_sequence: torch.Tensor) -> str:
        """Convert one-hot encoded sequence back to string."""
        sequence = encoded_sequence.reshape((self.config.sequence_length, len(AMINO_ACIDS)))
        idx = torch.argmax(sequence, dim=1)
        return "".join([AMINO_ACIDS[i] for i in idx])
    
    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Take action and return new state, reward, and done flag."""
        # Convert state to sequence and apply action
        sequence = self.decode_sequence(state)
        sequence_list = list(sequence)
        residue_idx = action // len(AMINO_ACIDS)
        new_aa = AMINO_ACIDS[action % len(AMINO_ACIDS)]
        sequence_list[residue_idx] = new_aa
        new_sequence = "".join(sequence_list)
        
        # Get new state and compute reward
        new_state = self.encode_sequence(new_sequence)
        pdb = infer_structure_batch([new_sequence])[0]
        stability = overall_confidence_from_pdb(pdb)
        reward = stability
        
        # Episode ends after max steps (can be modified)
        done = False
        
        return new_state, reward, done
    
    def reset(self) -> torch.Tensor:
        """Generate random initial sequence and return encoded state."""
        sequence = "".join(torch.randint(0, len(AMINO_ACIDS), (self.config.sequence_length,)).tolist())
        return self.encode_sequence(sequence)
