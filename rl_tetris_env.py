import time
import numpy as np
import pygame
from tetris_game import *
import gymnasium as gym
from gymnasium import spaces

USER = True


class RLTetrisEnv(Tetris):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces if needed for RL algorithms
        self.action_space = spaces.Discrete(44)

    def step(self, action):
        # the step function will be all of the time from the new peice starting at the top, then falling all the way and lines clearing as neccesary.

        num_rotations, num_movements = convert_to_action(action)  # unpack the action
        done = False  # initialize done

        # flatten the grid
        previous_board = convert_to_binary(self.grid)

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
                    unnecessary_movements = abs(
                        num_movements) - actual_movements
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
                board_before_clears = convert_to_binary(self.grid)
                rows_cleared = self.remove_full_rows()
                # Increase score for cleared rows
                self.score += self.calculate_game_score(rows_cleared)
                self.current_piece = self.new_piece()

                # Check for game over
                if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                    done = True

        # flatten the grid
        new_board = convert_to_binary(self.grid)
        flattened_grid = [float(item) for row in new_board for item in row]
        # Convert the peice number to a NumPy array
        next_shape_enum = SHAPES.index(self.current_piece['shape'])
        # Create a zero array of length equal to number of shapes
        one_hot_shape = [0] * len(SHAPES)
        # Set the index of the current piece to 1
        one_hot_shape[next_shape_enum] = 1

        # Add the one-hot encoded piece to the grid
        # Concatenate the flattened grid and one-hot encoded shape
        state = flattened_grid + one_hot_shape
        next_state = np.array(state, dtype=np.float32)

        # reward function
        reward = calculate_reward(previous_board, board_before_clears, new_board, rows_cleared)

        # end the game if the socre gets high enough
        if self.score > 400000:
            done = True

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
        self.grid = [[0 for _ in range(GRID_WIDTH)]
                     for _ in range(GRID_HEIGHT)]

        # Create the first tetromino piece
        self.current_piece = self.new_piece()

        # Game state variables
        self.game_over = False
        self.score = 0

        # Set up font for rendering text
        self.font = pygame.font.Font(None, 36)

        # Set up delay for continuous movement
        self.move_delay = 100  # Delay in milliseconds
        self.last_move_time = {pygame.K_LEFT: 0,
                               pygame.K_RIGHT: 0, pygame.K_DOWN: 0}

        # flatten the grid
        binary_grid = convert_to_binary(self.grid)
        flattened_grid = [float(item) for row in binary_grid for item in row]
        # Convert the peice number to a NumPy array
        next_shape_enum = SHAPES.index(self.current_piece['shape'])
        # Create a zero array of length equal to number of shapes
        one_hot_shape = [0] * len(SHAPES)
        # Set the index of the current piece to 1
        one_hot_shape[next_shape_enum] = 1

        # Add the one-hot encoded piece to the grid
        # Concatenate the flattened grid and one-hot encoded shape
        state = flattened_grid + one_hot_shape
        next_state = np.array(state, dtype=np.float32)
        info = ""

        return next_state, info
    
    def run_with_reward(self):
        fall_time = 0
        fall_speed = 0.5  # Time in seconds before the piece falls one block

        new_board = convert_to_binary(self.grid)

        while not self.game_over:
            fall_time += self.clock.get_rawtime()
            self.clock.tick()

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        # Rotate the piece if it's a valid move
                        rotated_piece = self.rotate_piece(self.current_piece)
                        if self.valid_move(rotated_piece, rotated_piece['x'], rotated_piece['y']):
                            self.current_piece = rotated_piece

            # Handle continuous movement (left, right, down)
            self.handle_continuous_movement()

            # Make the piece fall
            if fall_time / 1000 > fall_speed:
                if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                    self.current_piece['y'] += 1
                else:
                    # If the piece can't move down, place it and create a new piece
                    self.place_piece(self.current_piece)
                    board_before_clears = convert_to_binary(self.grid)
                    rows_cleared = self.remove_full_rows()
                    # Increase score for cleared rows
                    self.score += self.calculate_game_score(rows_cleared)

                    # update the boards and calc the reward
                    previous_board = new_board
                    new_board = convert_to_binary(self.grid)
                    reward = calculate_reward(previous_board, board_before_clears, new_board, rows_cleared)
                    print("Reward:")
                    print(reward)

                    self.current_piece = self.new_piece()

                    # Check for game over
                    if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                        self.game_over = True

                fall_time = 0  # Reset fall time

            # Draw the game state
            self.draw()

    def render(self):
        # create a render function to draw the board when desired
        self.draw()

    def get_state_size(self):
        return 201


# --------------HELPER FUNCTIONS--------------------------

def calculate_reward(previous_binary, before_clears_binary, new_binary, lines_cleared):

    # initialize the reward
    reward = 0

    # add increasing reward for each block i the same column
    # e.g. a horizontal I might yeild 8+7+6+5 if the row aleardy contained 4 filled blocks.
    _, _, cumulative_sums = count_new_blocks(previous_binary, before_clears_binary)
    reward += np.sum(cumulative_sums)

    # add a qudratic reward for clearing lines to incentivize combos
    reward += 10 * lines_cleared ** 2
    # 10, 40, 90, 160

    # reward based on placing peices that minimize unoccupied edges
    '''
    unoccupied_edges = calculate_edge_reward(previous_binary, before_clears_binary)
    if len(unoccupied_edges) < 1:
        reward += 0 # this shouldnt be possible
    else:
        reward += 10/len(unoccupied_edges)
        '''

    # encourage keeping the max column height low
    # max_column_height = calculate_highest_height(new_binary)
    # reward += 5 / max(max_column_height, 0.1) # dont divide by 0

    # encourage keeping the average column height low
    # avg_column_height = self.calculate_average_height(new_binary)
    # reward += 5 / avg_column_height

    # encourage filling all holes
    # unreachable_spaces = self.calculate_unreachable_spaces(new_binary)
    # reward += 5 / unreachable_spaces
    reward = np.float32(reward)

    return reward


def convert_to_binary(grid):
    def cell_to_binary(cell):
        if isinstance(cell, tuple):
            # If any RGB value is non-zero, consider it occupied
            return 1 if any(cell) else 0
        return 1 if cell else 0  # For non-tuple values, treat non-zero as occupied

    return [[cell_to_binary(cell) for cell in row] for row in grid]


def count_new_blocks(previous_board, new_board):
    new_blocks_count = []
    total_blocks_count = []
    cumulative_sums = []

    for prev_row, new_row in zip(previous_board, new_board):
        # Calculate the change in the row
        change = [new - prev for new, prev in zip(new_row, prev_row)]

        # Count the new blocks (number of 1s in the change)
        new_count = change.count(1)

        # Calculate the total number of blocks in the new row
        total_count = sum(new_row)

        # Calculate the cumulative sum based on total_count and new_count
        if new_count > 0:
            cumulative_sum = sum(range(total_count, total_count - new_count, -1))
        else:
            cumulative_sum = 0

        new_blocks_count.append(new_count)
        total_blocks_count.append(total_count)
        cumulative_sums.append(cumulative_sum)

    """
    print("Total Blocks in each row:")
    print(total_blocks_count)
    print("New Blocks in each row:")
    print(new_blocks_count)
    print("Cum Sums")
    print(cumulative_sums)

    time.sleep(5)
    """

    return new_blocks_count, total_blocks_count, cumulative_sums


def calculate_unreachable_spaces(binary_grid):
    # Assuming the grid is a standard Tetris board with width 10
    width = 10
    height = len(binary_grid) // width

    if len(binary_grid) != width * height:
        raise ValueError("Invalid grid size. Expected a multiple of 10.")

    def is_valid(x, y):
        return 0 <= x < height and 0 <= y < width

    def flood_fill(x, y):
        if not is_valid(x, y) or binary_grid[x * width + y] == 1 or visited[x * width + y]:
            return

        visited[x * width + y] = True

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            flood_fill(x + dx, y + dy)

    visited = [False] * len(binary_grid)

    # Fill from the top
    for j in range(width):
        if binary_grid[j] == 0:
            flood_fill(0, j)

    # Count unreachable holes
    unreachable_spaces = sum(1 for i in range(len(binary_grid))
                             if binary_grid[i] == 0 and not visited[i])

    return unreachable_spaces


def calculate_average_height(grid):
    width = len(grid[0])
    heights = [0] * width

    for col in range(width):
        for row in range(len(grid)):
            if grid[row][col] == 1:
                heights[col] = len(grid) - row
                break

    return sum(heights) / width

def calculate_edge_reward(previous_board, new_board):
    rows = len(new_board)
    cols = len(new_board[0])
    edge_reward = 0
    unoccupied_spaces = set()
    
    def count_unoccupied_edges(row, col):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if new_board[new_row][new_col] == 0:
                    unoccupied_spaces.add((new_row, new_col))
            else:
                # Count edges of the board as occupied
                pass

    for row in range(rows):
        for col in range(cols):
            if new_board[row][col] == 1 and previous_board[row][col] == 0:
                # This is a newly placed piece
                count_unoccupied_edges(row, col)

    return unoccupied_spaces


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
    state, reward = env.reset()
    print(state)
    print(reward)


    if USER:
        # env.run_with_reward()
        print("done")
    else:
        while not done:
            action = np.random.randint(0, 43)
            observation, reward, done, _, _ = env.step(action)
            env.render()
            time.sleep(3)  # Pauses for 1 second
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
