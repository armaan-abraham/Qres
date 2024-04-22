import numpy as np
import torch
import torch.nn as nn

from qres.protein import (
    AMINO_ACIDS,
    flattened_distance_matrix_length,
    flattened_quaternions_length,
)


def sequence_onehot_length(sequence_length):
    return sequence_length * len(AMINO_ACIDS)


def objective_length(sequence_length):
    return int(sequence_length * 2 + 1)


def action_length(sequence_length):
    return sequence_length * len(AMINO_ACIDS)


class DQN(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        # define embedder of protein state, which takes in the sequence, distance matrix, and quaternions
        sequence_oh_len = sequence_onehot_length(sequence_length)
        distances_len = flattened_distance_matrix_length(sequence_length)
        quaternions_len = flattened_quaternions_length(sequence_length)
        protein_state_len = int(sequence_oh_len + distances_len + quaternions_len)
        # protein_embedder_layer_dim = int(2 ** (int(np.log2(protein_state_len)) + 1))
        protein_embedder_layer_dim = 256
        print("embedder_layer_dim:", protein_embedder_layer_dim)
        self.protein_embedder = nn.Sequential(
            nn.Linear(protein_state_len, protein_embedder_layer_dim),
            nn.ReLU(),
            nn.Linear(protein_embedder_layer_dim, protein_embedder_layer_dim),
            nn.ReLU(),
            nn.Linear(protein_embedder_layer_dim, protein_embedder_layer_dim),
            nn.ReLU(),
            nn.Linear(protein_embedder_layer_dim, protein_embedder_layer_dim),
            nn.ReLU(),
        )

        # define the Q function
        action_len = action_length(sequence_length)
        self.q_function = nn.Linear(protein_embedder_layer_dim, action_len)

    def forward(self, protein_state):
        assert protein_state.shape[1] == self.protein_embedder[0].in_features
        protein_embedding = self.protein_embedder(protein_state)
        return self.q_function(protein_embedding)
