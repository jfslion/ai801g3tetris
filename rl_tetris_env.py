import time
import numpy as np
import pygame
from tetris_game import *
import gymnasium as gym
from gymnasium import spaces

class RLTetrisEnv(Tetris):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces if needed for RL algorithms
        self.action_space = spaces.Discrete(44)

    def step(self, action):
        # the step function will be all of the time from the new peice starting at the top, then falling all the way and lines clearing as neccesary.

        num_rotations, num_movements = convert_to_action(action) # unpack the action
        done = False # initialize done

        # Rotate the piece based on the number of rotations
        for _ in range(num_rotations):
            rotated_piece = self.rotate_piece(self.current_piece)
            if self.valid_move(rotated_piece, rotated_piece['x'], rotated_piece['y']):
               self.current_piece = rotated_piece

        # Move left or right based on num_movements
        unnecessary_movements = 0
        if num_movements != 0:
            direction = 1 if num_movements > 0 else -1
            actual_movements = 0
            for _ in range(abs(num_movements)):
                new_x = self.current_piece['x'] + direction
                if self.valid_move(self.current_piece, self.current_piece['x'] + direction, self.current_piece['y']):
                    self.current_piece['x'] += direction
                    actual_movements += 1
                else:
                    # Stop moving if the next move is not valid
                    # Calculate penalty for unnecessary movements
                    unnecessary_movements = abs(num_movements) - actual_movements
                    break

        # Move the peice down until it cant move anymore
        can_fall = True
        while can_fall:
            if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
            else:
                can_fall = False
                # If the piece can't move down, place it and create a new piece
                self.place_piece(self.current_piece)
                rows_cleared = self.remove_full_rows()
                self.score += rows_cleared * 100  # Increase score for cleared rows
                self.current_piece = self.new_piece()

                # Check for game over
                if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                    done = True

        # flatten the grid
        binary_grid = convert_to_binary(self.grid)
        flattened_grid = [float(item) for row in binary_grid for item in row]
        # Convert the peice number to a NumPy array
        next_shape_enum = [SHAPES.index(self.current_piece['shape'])]
        # add the peice enum to the grid
        state = flattened_grid + [float(next_shape_enum[0])]
        state_array = np.array(state, dtype=np.float32)

        next_state = state_array

        # reward function   
        rows_cleared_reward = rows_cleared^2
        game_over_penalty = int(done) * -1 
        height_managment_penalty = -calculate_average_height(binary_grid) + -calculate_highest_height(binary_grid)
        unnecessary_movements_penalty = -unnecessary_movements
        reward = rows_cleared_reward + game_over_penalty + height_managment_penalty + unnecessary_movements_penalty # to do: implement a function

        # include additional info
        info = ""
        truncated = False
        return next_state, reward, done, truncated, info
    
    def reset(self):
        # reset function is like an init function
        # Set up the game window
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        
        # Create a clock object to control the game's framerate
        self.clock = pygame.time.Clock()
        
        # Initialize the game grid (0 represents empty cells)
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Create the first tetromino piece
        self.current_piece = self.new_piece()
        
        # Game state variables
        self.game_over = False
        self.score = 0
        
        # Set up font for rendering text
        self.font = pygame.font.Font(None, 36)
        
        # Set up delay for continuous movement
        self.move_delay = 100  # Delay in milliseconds
        self.last_move_time = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0, pygame.K_DOWN: 0}

        # flatten the grid
        binary_grid = convert_to_binary(self.grid)
        flattened_grid = [float(item) for row in binary_grid for item in row]
        # Convert the peice number to a NumPy array
        next_shape_enum = [SHAPES.index(self.current_piece['shape'])]
        # add the peice enum to the grid
        state = flattened_grid + [float(next_shape_enum[0])]
        state_array = np.array(state, dtype=np.float32)

        next_state = state_array

        info = ""

        return next_state, info

    def render(self):
        # create a render function to draw the board when desired
        self.draw()
    
    def get_state_size(self):
        return 201

# help create a binary occupancy grid for the agent to analyze
def convert_to_binary(grid):
    def cell_to_binary(cell):
        if isinstance(cell, tuple):
            return 1 if any(cell) else 0  # If any RGB value is non-zero, consider it occupied
        return 1 if cell else 0  # For non-tuple values, treat non-zero as occupied

    return [[cell_to_binary(cell) for cell in row] for row in grid]

def calculate_average_height(grid):
    width = len(grid[0])
    heights = [0] * width
    
    for col in range(width):
        for row in range(len(grid)):
            if grid[row][col] == 1:
                heights[col] = len(grid) - row
                break
    
    return sum(heights) / width

def convert_to_action(n):
    if not 0 <= n <= 43:
        raise ValueError("Input must be between 0 and 43 inclusive")
    
    # First number (0 to 3)
    first = n // 11
    
    # Second number (-5 to 5)
    second = n % 11 - 5
    
    return first, second


def calculate_highest_height(grid):
    width = len(grid[0])
    heights = [0] * width
    
    for col in range(width):
        for row in range(len(grid)):
            if grid[row][col] == 1:
                heights[col] = len(grid) - row
                break
    return max(heights)

# Example usage of RLTetrisEnv
if __name__ == "__main__":
    env = RLTetrisEnv()
    
    done = False
    env.reset()
    
    while not done:
        action = np.random.randint(0, 43)
        observation, reward, done, _, _ = env.step(action)
        # env.render()
        # time.sleep(3)  # Pauses for 1 second        
        print(f"Action: {action}, Reward: {reward}, Done: {done}")