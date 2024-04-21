import math
import numpy as np
import random
import typing
import yaml
from itertools import count
from pathlib import Path
from collections import deque, namedtuple


import uuid
import torch
import torch.optim as optim
import torch.nn.functional as F

from qres.model import *
from qres.protein import *

COMMON_TRAINING_DIR = Path(__file__).parent / "training"

# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 128
# GAMMA is the discount factor as mentioned in the previous section
GAMMA = 0.99
# EPS_START is the starting value of epsilon
EPS_START = 0.9
# EPS_END is the final value of epsilon
EPS_END = 0.05
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 1000
# TAU is the update rate of the target network
TAU = 0.005
# LR is the learning rate of the ``AdamW`` optimizer
LR = 1e-4
NUM_AGENTS = 10
NUM_EPISODES = 600
SEQUENCE_LEN = 25
ACTION_LENGTH = action_length(SEQUENCE_LEN)
OBJECTIVE_LENGTH = objective_length(SEQUENCE_LEN)
REWARDS_LOOKBACK = 50
LOSS_TOL = 5e-2
REWARD_SUM_TOL = 1e-2
CHECKPOINT_INTERVAL = 1
SAVE_TO_DISK = False

if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda")
else:
    print("No GPU available")
    device = torch.device("cpu")

Objective = namedtuple("Objective", ("idx1", "idx2", "distance"))

Transition = namedtuple(
    # sequence as one-hot, action as one-hot, objective as onehot and float, reward as float, episode as int
    "Transition",
    ("state1", "action", "state2", "objective", "reward", "episode"),
)


def train():
    # create new uuid for this training session
    training_uuid = str(uuid.uuid4())
    print("TRAINING UUID:", training_uuid)

    # make new directory for this training session
    TRAINING_DIR = COMMON_TRAINING_DIR / training_uuid
    TRAINING_DIR.mkdir()
    PARAMETERS_DIR = TRAINING_DIR / "parameters"
    PARAMETERS_DIR.mkdir()

    # dump some metadata (such as date, time) about this run into a file in
    # training dir
    with open(TRAINING_DIR / "metadata.yaml", "w") as file:
        yaml.dump(
            {
                "num_agents": NUM_AGENTS,
                "num_episodes": NUM_EPISODES,
                "sequence_len": SEQUENCE_LEN,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay": EPS_DECAY,
                "tau": TAU,
                "lr": LR,
                "rewards_lookback": REWARDS_LOOKBACK,
                "loss_tol": LOSS_TOL,
                "reward_sum_tol": REWARD_SUM_TOL,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
            },
            file,
            default_flow_style=False,
            explicit_start=True,
        )

    policy_net = DQN(SEQUENCE_LEN).to(device)
    target_net = DQN(SEQUENCE_LEN).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    memory = ReplayMemory(10000)

    with torch.no_grad():
        # states for the agents
        sequences, objectives, states, pdbs = rand_initialize_episodes(NUM_AGENTS)
        loss = [
            get_objective_loss(pdb, objective)
            for pdb, objective in zip(pdbs, objectives)
        ]
        episode_ids = torch.arange(NUM_AGENTS)  # these will be incremented
        rewards_trajectory = [
            deque([], maxlen=REWARDS_LOOKBACK) for i in range(NUM_AGENTS)
        ]
        # for storing episode information long-term
        episode_trajectories = []
        for i in range(NUM_AGENTS):
            episode_trajectories.append(
                {
                    "objective": decode_objective(objectives[i]),
                    "trajectory": [],
                    "status": "incomplete",
                    "episode_id": episode_ids[i],
                },
            )
            episode_trajectories[i]["trajectory"].append(
                {
                    "sequence": sequences[i],
                    "loss": loss[i],
                    "reward": 0,
                }
            )
        n_steps_by_episode = torch.zeros(NUM_AGENTS)
        n_episodes_completed = 0

    for t in count():
        # this whole loop is torch no grad, besides the call to optimize_model
        with torch.no_grad():
            actions = torch.stack(
                [
                    select_action(
                        policy_net, states[i], objectives[i], n_steps_by_episode[i]
                    )
                    for i in range(NUM_AGENTS)
                ]
            )

            sequences_new = [act(states[i], actions[i]) for i in range(NUM_AGENTS)]
            pdbs_new = generate_pdbs(sequences_new)
            states_new = make_protein_states(sequences_new, pdbs_new)

            loss_new = [
                get_objective_loss(pdb, objective)
                for pdb, objective in zip(pdbs_new, objectives)
            ]
            rewards = [loss - loss_new for loss, loss_new in zip(loss, loss_new)]

            for i in range(NUM_AGENTS):
                rewards_trajectory[i].append(rewards[i])
                # save transition
                transition = Transition(
                    states[i],
                    actions[i],
                    states_new[i],
                    objectives[i],
                    rewards[i],
                    episode_ids[i],
                )
                memory.push(transition)
                write_transition_to_csv(
                    transition,
                    TRAINING_DIR / f"{episode_ids[i]}.csv",
                )

                if (
                    len(rewards_trajectory[i]) == REWARDS_LOOKBACK
                    and sum(rewards_trajectory[i]) < REWARD_SUM_TOL
                ) or loss_new[i] < LOSS_TOL:
                    if loss_new[i] < LOSS_TOL:
                        episode_trajectories[i]["status"] = "success"
                    else:
                        episode_trajectories[i]["status"] = "failure"
                    episode_trajectories[i]["trajectory"].append(
                        {
                            "sequence": sequences_new[i],
                            "loss": loss_new[i],
                            "reward": rewards[i],
                        }
                    )
                    n_episodes_completed += 1
                    # this agent is done its episode
                    sequences[i], objectives[i], states[i], pdbs[i] = (
                        rand_initialize_episode()
                    )
                    loss[i] = get_objective_loss(pdbs[i], objectives[i])
                    rand_initialize_episodes(1)
                    # create new id based on max id
                    episode_ids[i] = episode_ids.max() + 1
                    rewards_trajectory[i].clear()
                    n_steps_by_episode[i] = 0
                    n_episodes_completed += 1

                    episode_trajectories.append(
                        {
                            "objective": decode_objective(objectives[i]),
                            "trajectory": [],
                            "status": "incomplete",
                            "episode_id": episode_ids[i],
                        },
                    )
                    episode_trajectories[i]["trajectory"].append(
                        {
                            "sequence": sequences[i],
                            "loss": loss[i],
                            "reward": 0,
                        }
                    )
                else:
                    episode_trajectories[i]["trajectory"].append(
                        {
                            "sequence": sequences_new[i],
                            "loss": loss_new[i],
                            "reward": rewards[i],
                        }
                    )
                    # update current state for this agent
                    states[i] = states_new[i]
                    loss[i] = loss_new[i]
                    sequences[i] = sequences_new[i]
                    pdbs[i] = pdbs_new[i]
                    n_steps_by_episode[i] += 1

        # Perform one step of the optimization (on the policy network)
        optimize_model(memory, policy_net, target_net, optimizer)

        with torch.no_grad():
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if t % CHECKPOINT_INTERVAL == 0:
                print("Step:", t)
                # save model parameters
                save_model(
                    policy_net,
                    target_net,
                    optimizer,
                    PARAMETERS_DIR / f"{t}.pt",
                )
                average_reward = 0
                num_steps = 0
                for trajectory in episode_trajectories:
                    average_reward += sum(
                        [step["reward"] for step in trajectory["trajectory"]]
                    )
                    num_steps += len(trajectory["trajectory"])
                average_reward /= num_steps
                # define training summary
                training_summary = {
                    "num_episodes": n_episodes_completed,
                    "step": t,
                    "average_reward": average_reward,
                    "episode_trajectories": episode_trajectories,
                }
                with open(TRAINING_DIR / "summary", "a") as file:
                    yaml.dump(
                        training_summary,
                        file,
                        default_flow_style=False,
                        explicit_start=True,
                    )

            # check if our overall training is done
            if n_episodes_completed >= NUM_EPISODES:
                break


def optimize_model(memory, policy_net, target_net, optimizer):
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


class ReplayMemory:
    def __init__(self, capacity):
        self.memory_deque = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory_deque.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory_deque, batch_size)

    def __len__(self):
        return len(self.memory_deque)


def write_transition_to_csv(transition, path):
    with open(path, "a") as f:
        f.write(
            f"{transition.state1},{transition.action},{transition.state2},{transition.objective},{transition.reward},{transition.episode}\n"
        )


def decode_sequence(sequence: torch.Tensor) -> str:
    sequence = sequence.reshape((SEQUENCE_LEN, len(AMINO_ACIDS)))
    idx = torch.argmax(sequence, axis=1)
    return "".join([AMINO_ACIDS[i] for i in idx])


def act(state: torch.Tensor, action: torch.Tensor) -> str:
    """
    Edits the sequence in state according to the action, returns the string
    representation of the new sequence.

    Action is integer
    """
    new_sequence = list(decode_sequence(
        state.clone().detach()[: sequence_onehot_length(SEQUENCE_LEN)]
    ))
    residue_idx, new_amino_acid = action // len(AMINO_ACIDS), action % len(AMINO_ACIDS)
    new_sequence[residue_idx] = AMINO_ACIDS[new_amino_acid]
    return ''.join(new_sequence)


def select_action(
    model: torch.nn.Module,
    protein_state: torch.Tensor,
    objective: torch.Tensor,
    n_steps_episode: int,
) -> torch.Tensor:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * n_steps_episode / EPS_DECAY
    )
    if sample > eps_threshold:
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return model(protein_state, objective).max(dim=0).indices
    else:
        return torch.tensor(
            np.random.choice(action_length(SEQUENCE_LEN)),
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


def make_protein_states(sequences: typing.List[str], pdbs=None) -> torch.Tensor:
    pdbs = pdbs or generate_pdbs(sequences)
    sequences_enc = torch.cat(
        [encode_sequence(sequence)[None, :] for sequence in sequences]
    )
    distance_matrices = np.array(
        [flatten_distance_matrix(get_distance_matrix(pdb)) for pdb in pdbs]
    )
    # scale distances and quaternions appropriately
    distance_matrices = torch.tensor(distance_matrices, device=device).float()
    distance_matrices /= get_max_physical_protein_length_A(SEQUENCE_LEN)
    quaternionss = np.concatenate(
        [flatten_quaternions(compute_quaternions(pdb))[None, :] for pdb in pdbs]
    )
    quaternionss = torch.tensor(quaternionss, device=device).float()
    assert distance_matrices.shape == (
        len(sequences),
        flattened_distance_matrix_length(SEQUENCE_LEN),
    )
    assert quaternionss.shape == (
        len(sequences),
        flattened_quaternions_length(SEQUENCE_LEN),
    )
    assert sequences_enc.shape == (len(sequences), sequence_onehot_length(SEQUENCE_LEN))
    protein_states = torch.cat((sequences_enc, distance_matrices, quaternionss), dim=1)
    return protein_states


def encode_sequence(sequence: str) -> torch.Tensor:
    # one-hot encode sequence
    idx = torch.tensor(
        [AMINO_ACIDS.index(aa) for aa in sequence], dtype=torch.long, device=device
    )
    idx_oh = F.one_hot(idx, num_classes=len(AMINO_ACIDS)).float().flatten()
    # check shape
    assert idx_oh.shape == (len(sequence) * len(AMINO_ACIDS),)
    return idx_oh


def encode_action(action: int) -> torch.Tensor:
    # one-hot encode action
    action_oh = (
        F.one_hot(torch.tensor([action], device=device), num_classes=ACTION_LENGTH)
        .float()
        .flatten()
    )
    # check shape
    assert action_oh.shape == (ACTION_LENGTH,)
    return action_oh


def encode_objective(objective: Objective) -> torch.Tensor:
    # one-hot encode each idx
    idx1_oh = (
        F.one_hot(
            torch.tensor([objective.idx1], device=device), num_classes=SEQUENCE_LEN
        )
        .float()
        .flatten()
    )
    idx2_oh = (
        F.one_hot(
            torch.tensor([objective.idx2], device=device), num_classes=SEQUENCE_LEN
        )
        .float()
        .flatten()
    )
    # concatenate these vectors and append the distance
    objective_enc = torch.cat(
        (idx1_oh, idx2_oh, torch.tensor([objective.distance], device=device).float())
    )
    # check shape
    assert objective_enc.shape == (2 * SEQUENCE_LEN + 1,)
    return objective_enc


def encode_objectives(objectives: typing.List[Objective]) -> torch.Tensor:
    return torch.cat([encode_objective(objective) for objective in objectives])


def rand_initialize_sequence():
    # note that AMINO_ACIDS is a string
    # convert to list of characters
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


def rand_initialize_episode():
    # try to use the plural form of this function because structure prediction
    # is quicker in batches
    sequence = rand_initialize_sequence()
    objective = encode_objective(rand_initialize_objective())
    pdb = generate_pdbs([sequence])[0]
    state = make_protein_states([sequence], [pdb])[0]
    return sequence, objective, state, pdb


def save_model(policy_net, target_net, optimizer, path):
    torch.save(
        {
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
