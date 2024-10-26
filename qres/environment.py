import torch
import torch.nn.functional as F
import numpy as np
import random
from qres.structure_prediction import AMINO_ACIDS, N_AMINO_ACIDS, StructurePredictor
from qres.config import config
from jaxtyping import Float, Bool
from typing import Tuple

"""
TODO:
- step through action selection

"""

class Environment:
    def __init__(self):
        if not config.fake_structure_prediction:
            self.structure_predictor = StructurePredictor()
        self.states = torch.stack(
            [self._init_state() for _ in range(config.batch_size)]
        )
        assert self.states.shape == (
            config.batch_size,
            config.sequence_length * len(AMINO_ACIDS),
        )
        self.steps_done = torch.zeros(config.batch_size, device=config.device)

    def _init_state(self) -> Float[torch.Tensor, "(residue amino_acid)"]:
        sequence = torch.zeros(
            (config.sequence_length, N_AMINO_ACIDS), device=config.device
        )
        for i in range(config.sequence_length):
            sequence[i, random.randint(0, N_AMINO_ACIDS - 1)] = 1
        return sequence.reshape(config.sequence_length * N_AMINO_ACIDS)

    def get_states(self) -> Float[torch.Tensor, "batch (residue amino_acid)"]:
        return self.states

    def step(
        self,
        states: Float[torch.Tensor, "batch (residue amino_acid)"],
        actions: Bool[torch.Tensor, "batch (residue amino_acid)"],
    ) -> Tuple[
        Float[torch.Tensor, "batch (residue amino_acid)"],
        Float[torch.Tensor, "batch 1"],
        Bool[torch.Tensor, "batch 1"],
    ]:
        next_states = self.apply_actions(states, actions)

        rewards = self.calculate_rewards(next_states)

        dones = self.steps_done >= config.max_episode_length

        for i in range(config.batch_size):
            if dones[i]:
                self.steps_done[i] = 0
                self.states[i] = self._init_state()
            else:
                self.states[i] = next_states[i]

        self.validate_states(self.states)

        return next_states, rewards

    def apply_actions(
        self,
        states: Float[torch.Tensor, "batch (residue amino_acid)"],
        actions: Bool[torch.Tensor, "batch (residue amino_acid)"],
    ) -> Float[torch.Tensor, "batch (residue amino_acid)"]:
        states = states.view(config.batch_size, config.sequence_length, N_AMINO_ACIDS)
        actions = actions.view(config.batch_size, config.sequence_length, N_AMINO_ACIDS)
        new_states = states.clone()

        # Find positions where actions are 1
        batch_indices, seq_indices, _ = torch.where(actions == 1)

        # Update the states at those positions
        new_states[batch_indices, seq_indices] = actions[batch_indices, seq_indices].to(
            dtype=new_states.dtype
        )

        return new_states.view(
            config.batch_size, config.sequence_length * N_AMINO_ACIDS
        )

    def validate_states(
        self, states: Float[torch.Tensor, "batch (residue amino_acid)"]
    ):
        assert states.shape == (
            config.batch_size,
            config.sequence_length * N_AMINO_ACIDS,
        )
        assert (
            states.view(config.batch_size, config.sequence_length, N_AMINO_ACIDS).sum(
                dim=-1
            )
            == 1
        ).all()

    def decode_state(self, state: Float[torch.Tensor, "residue amino_acid"]) -> str:
        state = state.reshape((config.sequence_length, N_AMINO_ACIDS))
        idx = torch.argmax(state, axis=1)
        return "".join([AMINO_ACIDS[i] for i in idx])

    def calculate_rewards(
        self, states: Float[torch.Tensor, "batch (residue amino_acid)"]
    ) -> Float[torch.Tensor, "batch 1"]:
        sequences = [self.decode_state(state) for state in states]
        if config.fake_structure_prediction:
            confidences = torch.rand(config.batch_size, device=config.device)
        else:
            pdbs = self.structure_predictor.predict_structure(sequences)
            confidences = [
                self.structure_predictor.overall_confidence_from_pdb(pdb)
                for pdb in pdbs
            ]
        rewards = torch.tensor(confidences, device=config.device).unsqueeze(1)
        return rewards
