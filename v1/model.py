import numpy as np
import torch
import torch.nn as nn

from protein import AMINO_ACIDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_onehot_length(sequence_length):
    return sequence_length * len(AMINO_ACIDS)


def objective_length(sequence_length):
    return sequence_length * 2 + 1


def action_length(sequence_length):
    return sequence_length * len(AMINO_ACIDS)


class DQN(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        # define embedder of protein state, which takes in the sequence, distance matrix, and quaternions
        sequence_oh_len = sequence_onehot_length(sequence_length)
        distances_size = sequence_length**2 / 2 - sequence_length
        quaternions_size = (sequence_length - 1) * 4
        protein_state_size = sequence_oh_len + distances_size + quaternions_size
        protein_embedder_layer_dim = 2 ** (int(np.log2(protein_state_size)) + 1)
        print("embedder_layer_dim:", protein_embedder_layer_dim)
        self.protein_embedder = nn.Sequential(
            nn.Linear(protein_state_size, protein_embedder_layer_dim),
            nn.ReLU(),
            nn.Linear(protein_embedder_layer_dim, protein_embedder_layer_dim),
            nn.ReLU(),
        )

        objective_len = objective_length(sequence_length)

        # define embedder of objective
        self.objective_embedder = nn.Sequential(
            nn.Linear(objective_len, objective_len), nn.ReLU()
        )

        # jointly embed the protein state and the objective
        joint_embedder_layer_dim = 2 ** (
            int(np.log2(protein_embedder_layer_dim + objective_len)) + 1
        )
        print("joint_embedder_layer_dim:", joint_embedder_layer_dim)
        self.joint_embedder = nn.Sequential(
            nn.Linear(
                protein_embedder_layer_dim + objective_len, joint_embedder_layer_dim
            ),
            nn.ReLU(),
            nn.Linear(joint_embedder_layer_dim, joint_embedder_layer_dim),
            nn.ReLU(),
        )

        # define the Q function
        action_len = action_length(sequence_length)
        self.q_function = nn.Linear(joint_embedder_layer_dim, action_len)

    def forward(self, protein_state, objective):
        return self.q_function(
            self.joint_embedder(
                torch.cat(
                    [
                        self.protein_embedder(protein_state),
                        self.objective_embedder(objective),
                    ],
                    dim=1,
                )
            )
        )
