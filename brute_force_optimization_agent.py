# Import required libraries
import numpy as np
import time

import pygame
from rl_tetris_env import *

# Define a brute-force agent
class BruteForceAgent:
    def __init__(self, env):
        self.env = env

    def select_best_action(self):
        # Store the best action and reward found
        best_reward = -float('inf')
        best_action = None

        # Save the true state of the game
        original_state = self.env.copy_state()

        # Iterate over all possible actions to find the one with the highest reward
        for action in range(self.env.action_space.n):
            # Load the original state to avoid affecting the actual game state
            self.env.load_state(original_state)

            # Perform the action in the simulated state
            obs, reward, _, _, _ = self.env.step(action)

            # Update the best action if the reward is higher
            if reward > best_reward:
                best_reward = reward
                best_action = action

        # reset to original state
        self.env.load_state(original_state)        

        # Return the best action found after testing all options
        return best_action

    def step(self):
        # Select the best action based on simulated outcomes
        best_action = self.select_best_action()

        # Apply the best action to the actual environment
        _, reward, done, _, _ = self.env.step(best_action)
        print(best_action)
        print(reward)
        self.env.draw()
        time.sleep(1)
        return done


# Main execution
if __name__ == "__main__":
    # Initialize the environment and agent
    env = RLTetrisEnv()
    agent = BruteForceAgent(env)

    done = False
    while not done:

        print("loop")
        # Agent decides on the best action and takes a step
        done = agent.step()

        # Display the environment and sleep for visualization
        # agent.env.render()
        # pygame.time.delay(2000)  # Wait for 2 seconds
