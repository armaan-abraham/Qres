import torch
import random
from qres.structure_prediction import AMINO_ACIDS, N_AMINO_ACIDS, StructurePredictorPool
from qres.config import config
from jaxtyping import Float, Bool
from typing import Tuple
from qres.logger import logger

"""
TODO:
- step through action selection

"""

class Environment:
    def __init__(self, structure_predictor_pool=None):
        if not config.fake_structure_prediction:
            if structure_predictor_pool is None:
                # Default to using all available GPUs
                num_gpus = torch.cuda.device_count()
                device_ids = list(range(num_gpus))
                self.structure_predictor_pool = StructurePredictorPool(num_workers=num_gpus, device_ids=device_ids)
            else:
                self.structure_predictor_pool = structure_predictor_pool
        
        self.states = torch.stack(
            [self._init_state() for _ in range(config.batch_size)]
        ) 
        self.validate_states(self.states)
        
        self.steps_done = torch.zeros(config.batch_size, device=config.device)
    
    def _init_state(self) -> Float[torch.Tensor, "(seq residue amino)"]:
        sequence = torch.zeros(
            (config.sequence_length, N_AMINO_ACIDS), device=config.device
        )
        for i in range(config.sequence_length):
            sequence[i, random.randint(0, N_AMINO_ACIDS - 1)] = 1
        
        # The initial sequence is the same as the current sequence at the start
        initial_sequence = sequence.reshape(-1)
        current_sequence = sequence.clone().reshape(-1)
        
        # Concatenate the initial and current sequences to form the state
        state = torch.cat([initial_sequence, current_sequence], dim=0)
        return state
    
    def get_states(self) -> Float[torch.Tensor, "batch (seq residue amino)"]:
        return self.states.clone()
    
    def parse_seqs_from_states(
        self, state: Float[torch.Tensor, "batch (seq residue amino)"]
    ) -> Tuple[
        Float[torch.Tensor, "batch (residue amino)"],
        Float[torch.Tensor, "batch (residue amino)"],
    ]:
        # Split the state into initial and current sequences
        split_dim = config.sequence_length * N_AMINO_ACIDS
        init_seqs = state[:, :split_dim]
        seqs = state[:, split_dim:]
        return init_seqs, seqs

    def step(
        self,
        states: Float[torch.Tensor, "batch (seq residue amino)"],
        actions: Bool[torch.Tensor, "batch (residue amino)"],
    ) -> Tuple[
        Float[torch.Tensor, "batch (seq residue amino)"],
        Float[torch.Tensor, "batch 1"],
        Bool[torch.Tensor, "batch 1"],
    ]:
        init_seqs, seqs = self.parse_seqs_from_states(states)

        next_seqs = self.apply_actions(seqs, actions)

        next_states, rewards = self.seqs_to_states_rewards(init_seqs, next_seqs)
        
        dones = self.steps_done >= config.max_episode_length
        
        self.steps_done += 1
        self.steps_done[dones] = 0
        
        for i in range(config.batch_size):
            if dones[i]:
                self.states[i] = self._init_state()
            else:
                self.states[i] = next_states[i]
        
        self.validate_states(self.states)
        
        return self.states.clone(), rewards
    
    def apply_actions(
        self,
        seqs: Float[torch.Tensor, "batch (residue amino_acid)"],
        actions: Bool[torch.Tensor, "batch (residue amino_acid)"],
    ) -> Float[torch.Tensor, "batch (residue amino_acid)"]:
        seqs = seqs.view(config.batch_size, config.sequence_length, N_AMINO_ACIDS)
        actions = actions.view(config.batch_size, config.sequence_length, N_AMINO_ACIDS)
        next_seqs = seqs.clone()

        # Find the residue where the action is 1 and replace the corresponding
        # residue in the sequence
        batch_indices, residue_indices, _ = torch.where(actions == 1)
        next_seqs[batch_indices, residue_indices] = actions[batch_indices, residue_indices].to(
            dtype=next_seqs.dtype
        )

        return next_seqs.view(
            config.batch_size, config.sequence_length * N_AMINO_ACIDS
        )
    
    def validate_states(
        self, states: Float[torch.Tensor, "batch (seq residue amino)"]
    ):
        assert states.shape == (
            config.batch_size,
            config.state_dim,
        )
        total_states = states.view(config.batch_size, 2, config.sequence_length, N_AMINO_ACIDS)
        assert (
            total_states.sum(dim=-1) == 1
        ).all(), "All residues should have exactly one amino acid selected"
    
    def decode_seq(self, seq: Float[torch.Tensor, "(residue amino)"]) -> str:
        seq = seq.view(config.sequence_length, N_AMINO_ACIDS)
        idx = torch.argmax(seq, dim=1).cpu().numpy()
        return "".join([AMINO_ACIDS[i] for i in idx])

    def seqs_to_states_rewards(
        self, 
        init_seqs: Float[torch.Tensor, "batch (residue amino)"],
        seqs: Float[torch.Tensor, "batch (residue amino)"]
    ) -> Tuple[Float[torch.Tensor, "batch (seq residue amino)"], Float[torch.Tensor, "batch 1"]]:
        seqs_str = [self.decode_seq(seq) for seq in seqs]

        if config.fake_structure_prediction:
            confidences = torch.rand(config.batch_size, device=config.device)
        else:
            # Use predictor pool
            confidences_list = self.structure_predictor_pool.predict_structure(seqs_str)
            confidences = torch.tensor(confidences_list, device=config.device)

        # add penalty term for number of edits
        distance_penalty = config.distance_penalty_coeff * (init_seqs != seqs).float().mean(dim=1)
        rewards = confidences - distance_penalty
        logger.put(DistancePenalty=distance_penalty.mean().item(), Confidence=confidences.mean().item(), Reward=rewards.mean().item())

        next_states = torch.cat([init_seqs, seqs], dim=1)

        return next_states, rewards
