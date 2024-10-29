import random
from typing import Tuple

import torch
from jaxtyping import Bool, Float

from qres.config import AMINO_ACIDS, N_AMINO_ACIDS, config
from qres.logger import logger
from qres.structure_prediction import StructurePredictor

"""
TODO:
- step through action selection

"""


def validate_states(states: torch.Tensor):
    assert (
        states.shape[1] == config.state_dim
    ), f"States have incorrect shape {states.shape[1]} != {config.state_dim}"
    assert (
        states.dtype == config.state_dtype
    ), f"States have incorrect data type {states.dtype} != {config.state_dtype}"
    # Ensure all amino acid indices are valid
    assert (
        (states >= 0) & (states < N_AMINO_ACIDS)
    ).all(), "Invalid amino acid indices in states"


class Environment:
    def __init__(self, device: str):
        self.device = device
        logger.log_str(f"Initializing environment on device {device}")
        self.structure_predictor = StructurePredictor(device=device)
        logger.log_str("Initialized structure predictor")

        self.states = torch.stack(
            [self._init_state() for _ in range(config.structure_predictor_batch_size)]
        )
        validate_states(self.states)

        self.steps_done = torch.zeros(
            config.structure_predictor_batch_size, device=self.device
        )

    def _init_state(self) -> torch.Tensor:
        # Generate a random initial sequence of amino acid indices
        sequence = torch.randint(
            low=0,
            high=N_AMINO_ACIDS,
            size=(config.sequence_length,),
            dtype=config.state_dtype,
            device=self.device,
        )
        # At the start, the initial and current sequences are the same
        initial_sequence = sequence.clone()
        current_sequence = sequence.clone()

        # Concatenate the initial and current sequences to form the state
        state = torch.cat(
            [initial_sequence, current_sequence], dim=0
        )  # Shape: (2 * sequence_length,)
        return state

    def get_states(self) -> Bool[torch.Tensor, "batch (seq residue amino)"]:
        return self.states.clone()

    def parse_seqs_from_states(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split the state into initial and current sequences
        split_dim = config.sequence_length
        init_seqs = states[:, :split_dim]  # Shape: (batch_size, sequence_length)
        seqs = states[:, split_dim:]  # Shape: (batch_size, sequence_length)
        return init_seqs, seqs

    def step(
        self,
        states: torch.Tensor,
        actions: Bool[torch.Tensor, "batch (residue amino)"],
    ) -> Tuple[
        torch.Tensor,
        Float[torch.Tensor, "batch 1"],
    ]:
        init_seqs, seqs = self.parse_seqs_from_states(states)

        next_seqs = self.apply_actions(seqs, actions)

        next_states, rewards = self.seqs_to_states_rewards(init_seqs, next_seqs)

        dones = self.steps_done >= config.max_episode_length

        self.steps_done += 1
        self.steps_done[dones] = 0

        for i in range(config.structure_predictor_batch_size):
            if dones[i]:
                self.states[i] = self._init_state()
            else:
                self.states[i] = next_states[i]

        validate_states(self.states)

        return self.states.clone(), rewards

    def apply_actions(
        self,
        seqs: torch.Tensor,
        actions: Bool[torch.Tensor, "batch (residue amino_acid)"],
    ) -> torch.Tensor:
        actions = actions.view(
            config.structure_predictor_batch_size, config.sequence_length, N_AMINO_ACIDS
        )
        next_seqs = seqs.clone()

        # Find the residue where the action is 1 and replace the corresponding
        # residue in the sequence
        batch_indices, residue_indices, amino_indices = torch.where(actions == 1)
        next_seqs[batch_indices, residue_indices] = amino_indices.to(
            dtype=config.state_dtype
        )

        return next_seqs

    def decode_seq(self, seq: torch.Tensor) -> str:
        return "".join([AMINO_ACIDS[i] for i in seq.cpu().numpy()])

    def seqs_to_states_rewards(
        self,
        init_seqs: torch.Tensor,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Float[torch.Tensor, "batch 1"]]:
        seqs_str = [self.decode_seq(seq) for seq in seqs]

        if config.fake_structure_prediction:
            confidences = torch.rand(
                config.structure_predictor_batch_size, device=self.device
            )
        else:
            pdbs = self.structure_predictor.predict_structure(seqs_str)
            confidences = torch.tensor(
                [
                    self.structure_predictor.overall_confidence_from_pdb(pdb)
                    for pdb in pdbs
                ],
                device=self.device,
            )

        # add penalty term for number of edits
        distance_penalty = config.distance_penalty_coeff * (
            init_seqs != seqs
        ).float().mean(dim=1)
        rewards = confidences - distance_penalty
        logger.log_attrs(
            DistancePenalty=distance_penalty.mean().item(),
            Confidence=confidences.mean().item(),
            Reward=rewards.mean().item(),
        )

        next_states = torch.cat([init_seqs, seqs], dim=1)

        return next_states, rewards
