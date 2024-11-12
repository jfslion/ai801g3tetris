import time
import numpy as np
from tetris_game import Tetris
from gymnasium import spaces
from reward_center import RewardCenter

class TetrisEnv(Tetris):
    """"""
    def __init__(self, opts, reward_params):
        """"""
        super().__init__()

        # Initialize RewardCenter
        self.rewards = RewardCenter(reward_params, opts['print_reward_calc'], opts['publish_rewards'])

        # Define action and observation spaces if needed for RL algorithms
        ##TODO: calculate this number using the grid size and number of shapes.##
        self.action_space = spaces.Discrete(44, seed=opts['random_seed'])

        # Begin step counter.
        self.steps = 0

        # Define additional grids.
        self.grid = []
        self.prev_grid = []
        self.uncleared_grid = []

        # Misc and toggles.
        self.print_reward_calc = opts['print_reward_calc']
        self.debug_grid = opts['debug_grid']
        self.score_cutoff = opts['score_cutoff']


    def step(self, action):
        """
        Step function will be all of the time from the new piece starting at the top, 
        then falling all the way and lines clearing as neccesary.
        """
        self.steps += 1

        # Unpack the action.
        num_rotations, num_movements = self.convert_to_action(action)  
        done = False 

        # Flatten the grid.
        self.prev_grid = self.convert_to_binary()

        # Rotate the piece.
        for _ in range(num_rotations):
            rotated_piece = self.rotate_piece(self.current_piece)
            if self.valid_move(rotated_piece, rotated_piece['x'], rotated_piece['y']):
                self.current_piece = rotated_piece

        # Move left or right based on num_movements.
        unnecesary_movements = False
        if num_movements != 0:
            direction = 1 if num_movements > 0 else -1
            for _ in range(abs(num_movements)):
                if self.valid_move(self.current_piece, self.current_piece['x'] + direction, self.current_piece['y']):
                    self.current_piece['x'] += direction
                else:
                    if self.print_reward_calc:
                        print("Unnecesary horizontal movement")
                    unnecesary_movements = True
                    break

        # Move the peice down until it cant move anymore.
        can_fall = True
        while can_fall:
            if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
            else:
                can_fall = False
                # If the piece can't move down, place it and create a new piece.
                self.place_piece(self.current_piece)
                self.uncleared_grid = self.convert_to_binary()
                rows_cleared = self.remove_full_rows()
                # Increase score for cleared rows
                self.score += self.calculate_game_score(rows_cleared)
                self.current_piece = self.new_piece()

                # Check for game over.
                if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                    done = True

        # Flatten the grid.
        self.grid = self.convert_to_binary()
        flattened_grid = [float(item) for row in self.grid for item in row]

        # Convert the peice number to a NumPy array
        next_shape_enum = self.conf.SHAPES.index(self.current_piece['shape'])
        # Create a zero array of length equal to number of shapes
        one_hot_shape = [0] * len(self.conf.SHAPES)
        # Set the index of the current piece to 1
        one_hot_shape[next_shape_enum] = 1

        # Add the one-hot encoded piece to the grid
        # Concatenate the flattened grid and one-hot encoded shape
        state = flattened_grid + one_hot_shape
        next_state = np.array(state, dtype=np.float32)

        # Show what the reward function is dealing with.
        if self.debug_grid:
            print('New Grid:')
            for row in self.grid: print(f'\t{row}')

        # Calculate reward.
        reward, reward_meta = self.rewards.calc_reward(self.prev_grid, 
                                               self.uncleared_grid, 
                                               self.grid, 
                                               rows_cleared, 
                                               self.steps, 
                                               unnecesary_movements)

        # End the game if the score gets high enough.
        if self.score > self.score_cutoff:
            done = True
            print(f'Ending game at max score of {self.score_cutoff}')

        return next_state, reward, done, reward_meta


    def reset(self):
        """"""
        super().__init__()
        self.total_pieces = 0

        # Flatten the grid.
        self.grid = self.convert_to_binary()
        flattened_grid = [float(item) for row in self.grid for item in row]

        # Convert the peice number to a NumPy array
        next_shape_enum = self.conf.SHAPES.index(self.current_piece['shape'])

        # Create a zero array of length equal to number of shapes
        one_hot_shape = [0] * len(self.conf.SHAPES)

        # Set the index of the current piece to 1
        one_hot_shape[next_shape_enum] = 1

        # Add the one-hot encoded piece to the grid
        # Concatenate the flattened grid and one-hot encoded shape
        state = flattened_grid + one_hot_shape
        next_state = np.array(state, dtype=np.float32)
        info = ""

        return next_state, info


    def convert_to_binary(self):
        """
        Since the tetris board holds color information in the location of a peice,
        convert each color to an occupied bit for simplicity of calculations
        """
        def cell_to_binary(cell):
            if isinstance(cell, tuple):
                # If any RGB value is non-zero, consider it occupied
                return 1 if any(cell) else 0
            return 1 if cell else 0  # For non-tuple values, treat non-zero as occupied

        return [[cell_to_binary(cell) for cell in row] for row in self.board]


    def convert_to_action(self, n):
        """
        Convert the int from 0 to 43 into a pair of action,
        a rotation 0 to 3 and a horizontal movement to the right or left 5 spaces.
        Note that is may take less than 5 movements to get to the end.
        This will not affect the agent and theres potential to introduce action masking.
        """
        if not 0 <= n <= 43:
            raise ValueError("Input must be between 0 and 43 inclusive")

        # First number (0 to 3)
        first = n // 11

        # Second number (-5 to 5)
        second = n % 11 - 5

        return first, second


    def get_state_size(self):
        """
        TODO: Define how this is calculated.
        """
        return 201


    def render(self):
        """
        Render function to draw the board when desired.
        """
        self.draw()

    # def run_with_reward(self):
    #     fall_time = 0
    #     fall_speed = 0.5  # Time in seconds before the piece falls one block

    #     new_board = convert_to_binary(self.grid)

    #     while not self.game_over:
    #         fall_time += self.clock.get_rawtime()
    #         self.clock.tick()

    #         # Handle Pygame events
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 return
    #             if event.type == pygame.KEYDOWN:
    #                 if event.key == pygame.K_UP:
    #                     # Rotate the piece if it's a valid move
    #                     rotated_piece = self.rotate_piece(self.current_piece)
    #                     if self.valid_move(rotated_piece, rotated_piece['x'], rotated_piece['y']):
    #                         self.current_piece = rotated_piece

    #         # Handle continuous movement (left, right, down)
    #         self.handle_continuous_movement()

    #         # Make the piece fall
    #         if fall_time / 1000 > fall_speed:
    #             if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
    #                 self.current_piece['y'] += 1
    #             else:
    #                 # If the piece can't move down, place it and create a new piece
    #                 self.steps += 1
    #                 self.place_piece(self.current_piece)
    #                 board_before_clears = convert_to_binary(self.grid)
    #                 rows_cleared = self.remove_full_rows()
    #                 # Increase score for cleared rows
    #                 self.score += self.calculate_game_score(rows_cleared)

    #                 # update the boards and calc the reward
    #                 previous_board = new_board
    #                 new_board = convert_to_binary(self.grid)
    #                 reward, reward_meta = calculate_reward(previous_board, 
    #                                                        board_before_clears, 
    #                                                        new_board, 
    #                                                        rows_cleared, 
    #                                                        self.steps, 
    #                                                        0)
    #                 print("Reward:")
    #                 print(reward)

    #                 self.current_piece = self.new_piece()

    #                 # Check for game over
    #                 if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
    #                     self.game_over = True

    #             fall_time = 0  # Reset fall time

    #         # Draw the game state
    #         self.draw()