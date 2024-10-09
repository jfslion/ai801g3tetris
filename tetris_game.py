import random
import numpy as np

# Include any constants used by the class
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

class TetrisGame:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT):
        self.board = TetrisBoard(width, height)
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.game_over = False
        self.level = 0

    def _generate_piece(self):
        # use TetrisPeice to pick out a random peice from 1-7
        pass

    def move(self, column, rotation):
        # Drop the peice in the board
        pass

    def _calculate_score(self, lines_cleared):
        # basic scoring implementation
        if lines_cleared == 1:
            return 100 * self.level
        elif lines_cleared == 2:
            return 300 * self.level
        elif lines_cleared == 3:
            return 500 * self.level
        elif lines_cleared == 4:
            return 800 * self.level
        return 0

    def get_state(self):
        return {
            'board': self.board.get_state(),
            'current_piece': self.current_piece.shape if self.current_piece else None,
            'next_piece': self.next_piece.shape if self.next_piece else None,
            'score': self.score,
            'game_over': self.game_over,
            'level': self.level
        }
    
class TetrisBoard:
    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.current_piece = None

    def place_piece(self, piece, column, orentation):
        # Place a peice given a column and orientation
        pass
        
    def is_valid(self):
        # Return true if board is valid
        # May not need
        pass

    def clear_line(self):
        # Clear the row and move all higher rows down
        # Add an empty row to the top
        pass

    def is_game_over(self):
        # Check if the game is over, e.g. a peice 
        # is above the bounds of the game board
        pass

    def get_state(self):
        # return the state of the gameboard as a binary occupancy grid
        pass

    def __str__(self):
        # print the game board out to a string representation
        pass
    
    def step(self, action):
        # Allows the agent to take an action and affect the state of the game
        pass

    def reward(self, lines_cleared):
        # analyze the board and determine a score
        # passing the number of lines cleared to assist...?
        # Use the difference in the current score and the previous score...?
        pass

class TetrisPeice:
    def __init__(self):
        self.current_piece = self.random_piece()

    def random_piece(self):
        # use a rng and create a peice
        pass

    def rotate(self, peice_type, num_rotations):
        # return the new peice after a set amount of rotations
        pass


def main():
    game = TetrisGame()
    peices_played = 0
    while not game.game_over:
        peices_played += 1
        print(f"Score: {game.score}")
        print(game.board)
        print(f"Next piece: {game.next_piece.shape_name}")


        column = input("Enter the column 1:10: ").strip().upper()
        orientation = input("Enter the orientation 1:4: ").strip().upper()

        if peices_played == 10:
            # increase the level every 10 peices played
            game.level += 1
            peices_played = 0
            


    # The game is now over
    print("Game Over!")
    print(f"Final Score: {game.score}")

if __name__ == "__main__":
    main()