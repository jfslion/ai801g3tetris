# -*- coding: utf-8 -*-
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from datetime import *
import glob
import inspect
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tkinter as tk
from tkinter import filedialog

import os

from env_tetris import TetrisEnv

env = TetrisEnv()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


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

TRAIN_AGENT = True
WATCH_REPLAY = not TRAIN_AGENT

EPISODES = 1000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
episode_rewards = []


def plot_durations(show_result=False):
    # Plot the Episode Durations in the first figure
    # Use the first figure (if already created, it won't open a new one)
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()  # Clear the current figure
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100-episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.legend([])  # Clear any previous legend
    plt.legend(["Episode Durations", "100-Episode Average"])  # Set new legend

    # Plot the Episode Rewards in the second figure
    plt.figure(2)  # Use the second figure
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy(), label="Reward per Episode", color='orange')

    # Take 100-episode averages and plot them too
    if len(rewards_t) >= 100:
        reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        reward_means = torch.cat((torch.zeros(99), reward_means))
        plt.plot(reward_means.numpy(),
                 label="100-episode Average Reward", color='red')

    plt.legend([])  # Clear any previous legend
    # Set new legend
    plt.legend(["Reward per Episode", "100-Episode Average Reward"])

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Directory to save/load model
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "tetris_dqn_model.pth")

# Define a save function


def save_model(model, reward_function=calculate_reward, filename_prefix='model'):
    # Get the current directory (where the script is running)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the current time and format it as a string
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a new filename with the prefix and current time
    model_filename = f"{filename_prefix}_{current_time}.pt"
    reward_filename = f"{filename_prefix}_reward_{current_time}.txt"

    # Create the full path for saving the model
    model_full_path = os.path.join(current_dir, model_filename)
    reward_full_path = os.path.join(current_dir, reward_filename)

    # Save the model state dict
    torch.save(model.state_dict(), model_full_path)
    print(f"Model saved to {model_full_path}")

    # Automatically get the source code of the reward function
    reward_content = inspect.getsource(reward_function)

    # Save the reward content to a text file
    with open(reward_full_path, 'w') as file:
        file.write(reward_content)
    print(f"Reward content saved to {reward_full_path}")


"""
# Define a load function
def load_model(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode for inference
    print(f"Model loaded from {path}")
"""

"""
def load_model(model):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select model file",
        filetypes=[("PyTorch Model", "*.pth")]
    )

    if file_path:
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set to evaluation mode for inference
        print(f"Model loaded from {file_path}")
    else:
        print("No file selected")

    root.destroy()  # Clean up the tkinter instance
    """


def load_model(model):
    # Get the directory where the models are saved
    model_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all model files matching the pattern
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))

    if not model_files:
        print("No model files found.")
        return

    # Sort the files by modification time, most recent first
    most_recent_model = max(model_files, key=os.path.getmtime)

    print(f"Loading most recent model: {most_recent_model}")

    # Load the model
    model.load_state_dict(torch.load(most_recent_model))
    model.eval()  # Set to evaluation mode for inference

    print(f"Model loaded successfully.")

    # Function to watch the agent play the game


def watch_agent_play(env, model, num_episodes=1):
    for episode in range(num_episodes):
        state, _ = env.reset()
        for t in count():
            env.render()  # Visualize the game
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0).to(device)
            action = model(state_tensor).max(1)[1].view(1, 1)
            state, reward, done, truncated, _ = env.step(action.item())
            env.render
            print(f"Action: {action.item()}, Reward: {reward}, Done: {done}")
            time.sleep(2)
            if done:
                break


if TRAIN_AGENT:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = EPISODES
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        episode_reward = 0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(
                action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            episode_reward += reward

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(episode_reward)
                plot_durations()
                save_interval = 500
                if i_episode % save_interval == 0:  # e.g., save every 50 episodes
                    save_model(policy_net)
                break

    print('Complete')
    # Save model after training
    save_model(policy_net)

    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if WATCH_REPLAY:
    # Load the model and watch it play
    load_model(policy_net)
    watch_agent_play(env, policy_net)
