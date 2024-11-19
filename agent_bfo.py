# Import required libraries
import time
import copy

# Define a brute-force agent
class BruteForceAgent:
    def __init__(self, env, display_choice = False, display_each = False):
        self.env = env
        self.display_each = display_each
        self.display_choice = display_choice

    def select_best_action(self):
        # Store the best action and reward found
        best_reward = -float('inf')
        best_action = None

        # Save the true state of the game
        original_state = self.env.copy_state()
        current_piece = copy.deepcopy(original_state['current_piece'])

        # Iterate over all possible actions to find the one with the highest reward.
        for action in range(self.env.action_space.n):
            # Load the original state to avoid affecting the actual game state.
            self.env.load_state(original_state)

            # Perform the action in the simulated state
            obs, reward, _, reward_meta, action_meta = self.env.step(action)
            self.env.load_current_piece(original_state['current_piece'])

            if self.display_each:
                self.env.draw(reward_meta, action_meta, 1000)

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
        _, reward, done, reward_meta, action_meta = self.env.step(best_action)

        if self.display_choice:
            self.env.draw(reward_meta, action_meta, 500)
            print(f'\nStep num: {self.env.steps}')
            print(f'Action Data: {action_meta}')
            print(f'Reward Data: {reward_meta}')
        return done, reward_meta, action_meta
