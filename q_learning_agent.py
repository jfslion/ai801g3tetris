import numpy as np
import random
from rl_tetris_env import *


class QLearningAgent:
    def __init__(self, state_size, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        # Convert the state to a tuple to use it as a key in the Q-table
        # Flatten the array to 1D and convert to tuple
        state_key = tuple(state.flatten())

        # Check if the state_key is in the Q-table
        if state_key not in self.q_table:
            # Initialize the Q-value for the state if not already present
            self.q_table[state_key] = {action: 0 for action in range(
                self.action_space.n)}  # Initialize all actions to 0

        return self.q_table[state_key]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action from the action space
            return self.action_space.sample()
        else:
            state_key = self.state_to_key(state)  # Convert the state to a key
            if state_key in self.q_table:
                # Choose the action with the highest Q-value
                return max(self.q_table[state_key], key=self.q_table[state_key].get)
            else:
                # If the state is not in the Q-table, sample a random action
                return self.action_space.sample()

    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_q_value(state, action)
        if not done:
            next_q = max(self.get_q_value(next_state, a)
                         for a in self.get_possible_actions(next_state))
            target_q = reward + self.discount_factor * next_q
        else:
            target_q = reward
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[self.state_to_key(state)][action] = new_q

    def state_to_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        else:
            return state  # Assuming it's already hashable

    def get_possible_actions(self, state):
        return [tuple(space.sample() for space in self.action_space.spaces)
                for _ in range(100)]  # Sample a reasonable number of


# Example usage:
env = RLTetrisEnv()  # Your Tetris environment
state_size = env.get_state_size()
action_space = env.action_space
RENDER = True

agent = QLearningAgent(state_size, action_space)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = agent.state_to_key(state)
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        if RENDER:
            env.render()
        next_state = agent.state_to_key(next_state)

        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    print(f"Episode {
          episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
