# Leverage the code from github repository: https://github.com/tucna/Programming-Basic-Concepts
# Utilize concepts from the follwoing video: https://www.youtube.com/watch?v=gIjVwODrXC8
import tetris_game_config as conf
import random
import copy
import numpy as np
from gymnasium import spaces
from reward_center import RewardCenter

class TetrisEnvNoGame:
    def __init__(self, opts, reward_params):
        # Save the configuration requested.
        self.conf = conf        

        # Initialize the game grid (0 represents empty cells).
        self.board = [[0 for _ in range(conf.GRID_WIDTH)]
                     for _ in range(conf.GRID_HEIGHT)]

        # Create the first tetromino piece.
        self.current_piece = self.new_piece()

        # Game state variables.
        self.game_over = False
        self.score = 0

        # Set up delay for continuous movement.
        self.move_delay = 100  # Delay in milliseconds

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


    def new_piece(self):
        # Randomly select a shape and its corresponding color.
        shape = random.choice(self.conf.SHAPES)
        color = self.conf.COLORS[self.conf.SHAPES.index(shape)]

        # Return a dictionary representing the new piece.
        return {
            'shape': shape,
            'color': color,
            # Center the piece horizontally
            'x': self.conf.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0  # Start at the top of the grid
        }

    def calculate_game_score(self, lines_cleared):
        if lines_cleared == 0:
            return 0

        points = {
            1: 40,    # Single
            2: 100,   # Double
            3: 300,   # Triple
            4: 1200   # Tetris
        }

        # Calculate score for this turn
        score = points[lines_cleared]
        return score

    def valid_move(self, piece, x, y):
        # Check if the piece can be placed at the given position
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    if (x + j < 0 or x + j >= self.conf.GRID_WIDTH or  # Check horizontal boundaries
                        y + i >= self.conf.GRID_HEIGHT or  # Check bottom boundary
                            # Check collision with placed pieces
                            (y + i >= 0 and self.board[y + i][x + j])):
                        return False
        return True

    def place_piece(self, piece):
        # Place the piece on the grid
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.board[piece['y'] + i][piece['x'] + j] = piece['color']

    def remove_full_rows(self):
        # Identify and remove full rows, then add new empty rows at the top
        full_rows = [i for i, row in enumerate(self.board) if all(row)]
        for row in full_rows:
            del self.board[row]
            self.board.insert(0, [0 for _ in range(self.conf.GRID_WIDTH)])
        return len(full_rows)  # Return the number of rows removed

    def rotate_piece(self, piece):
        # Rotate the piece 90 degrees clockwise
        return {
            # Transpose and reverse the shape matrix
            'shape': list(zip(*reversed(piece['shape']))),
            'color': piece['color'],
            'x': piece['x'],
            'y': piece['y']
        }

    def copy_state(self):
        """
        Return a dictionary with only serializable attributes
        """
        return {
            'grid': copy.deepcopy(self.grid),
            'current_piece': copy.deepcopy(self.current_piece),
            'game_over': self.game_over,
            'score': self.score,
            'steps': self.steps,
            'board' : copy.deepcopy(self.board),
        }


    def load_state(self, state):
        """
        Restore environment state from the dictionary
        """

        self.grid = copy.deepcopy(state['grid'])
        self.current_piece = copy.deepcopy(state['current_piece'])
        self.game_over = state['game_over']
        self.score = state['score']
        self.steps = state['steps']
        self.board = copy.deepcopy(state['board'])

    def load_current_piece(self, cur_piece):
        self.current_piece = copy.deepcopy(cur_piece)

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
        
        action_meta = { 'CurPiece' :     self.current_piece['shape'],
                        'ActionNum':     action,
                        'NumRotate':     num_rotations,
                        'NumHoriz':      num_movements,
        }

        unnecesary_movements = False
        # Rotate the piece.
        for _ in range(num_rotations):
            rotated_piece = self.rotate_piece(self.current_piece)
            if self.valid_move(rotated_piece, rotated_piece['x'], rotated_piece['y']):
                self.current_piece = rotated_piece
            else:
                unnecesary_movements = True
                # print("Piece rotation failure")

        # Move left or right based on num_movements.
        if num_movements != 0:
            direction = 1 if num_movements > 0 else -1
            for _ in range(abs(num_movements)+1):
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

        # Convert the piece number to a NumPy array
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

        return next_state, reward, done, reward_meta, action_meta


    def reset(self):
        """
        """
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
        return 207
