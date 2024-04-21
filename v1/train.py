import math
import numpy as np
import random
import typing
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import *
from protein import *

# TODO: check if we need to use torch.no_grad() anywhere

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_AGENTS = 10
NUM_EPISODES = 1000
SEQUENCE_LEN = 25
ACTION_LENGTH = action_length(SEQUENCE_LEN)
OBJECTIVE_LENGTH = objective_length(SEQUENCE_LEN)
REWARDS_LOOKBACK = 50
LOSS_TOL = 5e-2
REWARD_SUM_TOL = 1e-2

Objective = namedtuple("Objective", ("idx1", "idx2", "distance"))

# TODO: save the state better
Transition = namedtuple(
    # sequence as one-hot, action as one-hot, objective as onehot and float, reward as float, episode as int
    "Transition",
    ("state1", "action", "state2", "objective", "reward", "episode"),
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def decode_sequence(sequence: torch.Tensor) -> str:
    sequence = sequence.reshape((SEQUENCE_LEN, len(AMINO_ACIDS)))
    idx = torch.argmax(sequence, axis=1)
    return "".join([AMINO_ACIDS[i] for i in idx])


def act(state: torch.Tensor, action: torch.Tensor) -> str:
    new_sequence = state.copy()[: sequence_onehot_length()].reshape(
        (SEQUENCE_LEN, len(AMINO_ACIDS))
    )
    action = action.reshape((SEQUENCE_LEN, len(AMINO_ACIDS)))
    residue_idx, new_amino_acid = torch.where(action)
    new_sequence[residue_idx] = AMINO_ACIDS[new_amino_acid]
    return decode_sequence(new_sequence)


def select_action(protein_state, objective):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(protein_state, objective).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[np.random.choice(action_length(SEQUENCE_LEN))]],
            device=device,
            dtype=torch.long,
        )


def decode_objective(objective: torch.Tensor) -> Objective:
    idx1 = torch.argmax(objective[:SEQUENCE_LEN])
    idx2 = torch.argmax(objective[SEQUENCE_LEN : 2 * SEQUENCE_LEN])
    distance = objective[-1]
    return Objective(idx1=idx1, idx2=idx2, distance=distance)


def get_objective_loss(pdb, objective):
    max_length = get_max_physical_protein_length_A(SEQUENCE_LEN)
    objective = decode_objective(objective)
    dm = get_distance_matrix(pdb)
    dist = dm[objective.idx1][objective.idx2]
    return (dist / max_length - objective.distance) ** 2

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    states1 = torch.cat(batch.state1)
    states2 = torch.cat(batch.state2)
    objectives = torch.cat(batch.objective)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    Q = policy_net(states1, objectives).gather(1, actions).squeeze()

    with torch.no_grad():
        V_next = target_net(states2, objectives).max(1).values

    # Bellman equation
    Q_expected = (V_next * GAMMA) + rewards

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(Q, Q_expected)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def make_protein_states(sequences: typing.List[str], pdbs=None) -> torch.Tensor:
    pdbs = pdbs or generate_pdbs(sequences)
    sequences_enc = torch.cat(
        [encode_sequence(sequence) for sequence in sequences], axis=0
    )
    distance_matrices = [
        flatten_distance_matrix(get_distance_matrix(pdb)) for pdb in pdbs
    ]
    quaternionss = [flatten_quaternions(compute_quaternions(pdb)) for pdb in pdbs]
    # scale distances and quaternions appropriately
    distance_matrices = torch.tensor(distance_matrices).float()
    distance_matrices /= get_max_physical_protein_length_A(SEQUENCE_LEN)
    quaternionss = torch.tensor(quaternionss).float()
    protein_states = torch.cat(sequences_enc, distance_matrices, quaternionss, axis=1)
    return protein_states


def encode_sequence(sequence: str) -> torch.Tensor:
    # one-hot encode sequence
    idx = torch.tensor([AMINO_ACIDS.index(aa) for aa in sequence], dtype=torch.long)
    idx_oh = F.one_hot(idx, num_classes=len(AMINO_ACIDS)).float().flatten()
    # check shape
    assert idx_oh.shape == (len(sequence) * len(AMINO_ACIDS),)
    return idx_oh


def encode_action(action: int) -> torch.Tensor:
    # one-hot encode action
    action_oh = (
        F.one_hot(torch.tensor([action]), num_classes=ACTION_LENGTH).float().flatten()
    )
    # check shape
    assert action_oh.shape == (ACTION_LENGTH,)
    return action_oh


def encode_objective(objective: Objective) -> torch.Tensor:
    # one-hot encode each idx
    idx1_oh = (
        F.one_hot(torch.tensor([objective.idx1]), num_classes=SEQUENCE_LEN)
        .float()
        .flatten()
    )
    idx2_oh = (
        F.one_hot(torch.tensor([objective.idx2]), num_classes=SEQUENCE_LEN)
        .float()
        .flatten()
    )
    # concatenate these vectors and append the distance
    objective_enc = torch.cat(
        (idx1_oh, idx2_oh, torch.tensor([objective.distance]).float())
    )
    # check shape
    assert objective_enc.shape == (2 * SEQUENCE_LEN + 1,)
    return objective_enc


def encode_objectives(objectives: typing.List[Objective]) -> torch.Tensor:
    return torch.cat([encode_objective(objective) for objective in objectives], axis=0)


def rand_initialize_sequence():
    return "".join(np.random.choice(AMINO_ACIDS, size=SEQUENCE_LEN))


def rand_initialize_objective():
    idx1 = np.random.choice(SEQUENCE_LEN - 2)
    idx2 = np.random.choice(np.arange(idx1 + 2, SEQUENCE_LEN))
    max_length = get_max_physical_protein_length_A(SEQUENCE_LEN)
    return Objective(
        idx1=idx1,
        idx2=idx2,
        distance=np.random.uniform(
            5 / max_length, (idx2 - idx1) * AMINO_ACID_LENGTH_ANGSTROM / max_length
        ),
    )


def rand_initialize_episodes(num_episodes: int):
    sequences = [rand_initialize_sequence() for i in range(num_episodes)]
    objectives = [
        encode_objective(rand_initialize_objective()) for i in range(num_episodes)
    ]
    pdbs = generate_pdbs(sequences)
    states = make_protein_states(sequences, pdbs)
    return sequences, objectives, states, pdbs


policy_net = DQN(SEQUENCE_LEN).to(device)
target_net = DQN(SEQUENCE_LEN).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

sequences, objectives, states, pdbs = rand_initialize_episodes(NUM_AGENTS)
loss = [get_objective_loss(pdb, objective) for pdb, objective in zip(pdbs, objectives)]
episode_ids = torch.arange(NUM_AGENTS)  # these will be incremented
rewards_trajectory = [deque([], maxlen=REWARDS_LOOKBACK) for i in NUM_AGENTS]


for t in count():
    actions = [select_action(states[i], objectives[i]) for i in NUM_AGENTS]

    sequences_new = [act(states[i], actions[i]) for i in NUM_AGENTS]
    pdbs_new = generate_pdbs(sequences_new)
    states_new = torch.tensor(
        make_protein_states(sequences_new, pdbs_new), device=device
    )

    loss_new = [get_objective_loss(pdb, objective) for pdb, objective in zip(pdbs_new, objectives)]
    rewards = [loss - loss_new for loss, loss_new in zip(loss, loss_new)]

    for i in NUM_AGENTS:
        rewards_trajectory[i].append(rewards[i])
        memory.push(states[i], actions[i], states_new[i], objectives[i], rewards[i])
        if (len(rewards_trajectory) == REWARDS_LOOKBACK and sum(rewards_trajectory[i]) < REWARD_SUM_TOL) or loss_new[i] < LOSS_TOL:

            sequences[i], objectives[i], states[i], pdbs[i] = rand_initialize_episodes(1)
            episode_ids[i] += 1
            rewards_trajectory[i].clear()




    

    states = states_new

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

