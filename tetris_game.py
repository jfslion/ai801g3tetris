# Leverage the code from github repository: https://github.com/tucna/Programming-Basic-Concepts
# Utilize concepts from the follwoing video: https://www.youtube.com/watch?v=gIjVwODrXC8
import pygame
import tetris_game_config as conf
import random

class Tetris:
    def __init__(self):
        # Save the configuration requested.
        self.conf = conf        

        # Initialize pygame
        pygame.init()

        # Set up the game window.
        self.screen = pygame.display.set_mode((conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")

        # Create a clock object to control the game's framerate.
        self.clock = pygame.time.Clock()

        # Initialize the game grid (0 represents empty cells).
        self.board = [[0 for _ in range(conf.GRID_WIDTH)]
                     for _ in range(conf.GRID_HEIGHT)]

        # Create the first tetromino piece.
        self.current_piece = self.new_piece()

        # Game state variables.
        self.game_over = False
        self.score = 0

        # Set up font for rendering text.
        self.font = pygame.font.Font(None, 36)

        # Set up delay for continuous movement.
        self.move_delay = 100  # Delay in milliseconds
        self.last_move_time = {pygame.K_LEFT: 0,
                               pygame.K_RIGHT: 0, pygame.K_DOWN: 0}

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

    def draw_border(self):
        # Draw the border around the game area
        pygame.draw.rect(
            self.screen, self.conf.GRAY, (0, 0, self.conf.SCREEN_WIDTH - 200, self.conf.SCREEN_HEIGHT), self.conf.BORDER_WIDTH)

    def draw(self, reward_meta = None, action_meta = None, delay = None):
        # Clear the screen
        self.screen.fill(self.conf.BLACK)

        # Draw the border
        self.draw_border()

        # Draw the placed pieces on the grid.
        for y, row in enumerate(self.board):
            for x, color in enumerate(row):
                if color:
                    pygame.draw.rect(self.screen, color,
                                     (x * self.conf.BLOCK_SIZE + self.conf.BORDER_WIDTH,
                                      y * self.conf.BLOCK_SIZE,
                                      self.conf.BLOCK_SIZE - 1, self.conf.BLOCK_SIZE - 1))

        # Draw the current falling piece.
        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.current_piece['color'],
                                     ((self.current_piece['x'] + j) * self.conf.BLOCK_SIZE + self.conf.BORDER_WIDTH,
                                      (self.current_piece['y'] +
                                       i) * self.conf.BLOCK_SIZE,
                                      self.conf.BLOCK_SIZE - 1, self.conf.BLOCK_SIZE - 1))

        # Initialize the text height.
        text_height = 10
        text_block_sep = 10*3

        # Display the Score.
        self.font = pygame.font.Font(None, 36)
        score_text = self.font.render(f'Score: {self.score}', True, self.conf.WHITE)
        self.screen.blit(score_text, (self.conf.SCREEN_WIDTH - 190, text_height))

        if action_meta:
            text_height += text_block_sep
            self.font = pygame.font.Font(None, 18)
            for indx, (key, val) in enumerate(action_meta.items()):
                if isinstance(val, list):
                    temp_text = f'{key}:'
                    for num, line in enumerate(val):
                        if num == 0:
                            temp_text = f'{key}: {line}'
                        else:
                            temp_text = f'                     {line}'
                        text_height += 10
                        action_text = self.font.render(f'{temp_text}', True, self.conf.WHITE)
                        self.screen.blit(action_text, (self.conf.SCREEN_WIDTH - 190, text_height))
                else:
                    action_text = self.font.render(f'{key}: {val}', True, self.conf.WHITE)
                    text_height += 15
                    self.screen.blit(action_text, (self.conf.SCREEN_WIDTH - 190, text_height))

        # Print the reward calcs if present.
        if reward_meta:
            text_height += text_block_sep
            self.font = pygame.font.Font(None, 18)
            for indx, (rew, val) in enumerate(reward_meta.items()):
                score_text = self.font.render(f'{rew}: {val:.2f}', True, self.conf.WHITE)
                text_height += 15
                self.screen.blit(score_text, (self.conf.SCREEN_WIDTH - 190, text_height))

        # Draw game over message if the game has ended.
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, self.conf.WHITE)
            self.screen.blit(game_over_text, (self.conf.SCREEN_WIDTH //
                             2 - 70, self.conf.SCREEN_HEIGHT // 2))
            print(f'Final Score: {self.score}')

        # Update the display.
        pygame.display.flip()
        if delay:
            self.delay_game(delay)

    def handle_continuous_movement(self):
        # Handle continuous key presses for smoother movement
        keys = pygame.key.get_pressed()
        current_time = pygame.time.get_ticks()

        # Move left
        if keys[pygame.K_LEFT] and current_time - self.last_move_time[pygame.K_LEFT] > self.move_delay:
            if self.valid_move(self.current_piece, self.current_piece['x'] - 1, self.current_piece['y']):
                self.current_piece['x'] -= 1
                self.last_move_time[pygame.K_LEFT] = current_time

        # Move right
        if keys[pygame.K_RIGHT] and current_time - self.last_move_time[pygame.K_RIGHT] > self.move_delay:
            if self.valid_move(self.current_piece, self.current_piece['x'] + 1, self.current_piece['y']):
                self.current_piece['x'] += 1
                self.last_move_time[pygame.K_RIGHT] = current_time

        # Move down
        if keys[pygame.K_DOWN] and current_time - self.last_move_time[pygame.K_DOWN] > self.move_delay:
            if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
                self.last_move_time[pygame.K_DOWN] = current_time


    def delay_game(self, d_time):
        """
        """
        pygame.time.delay(d_time)

    def run(self):
        fall_time = 0
        fall_speed = 0.5  # Time in seconds before the piece falls one block

        while not self.game_over:
            fall_time += self.clock.get_rawtime()
            self.clock.tick()

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
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
                    rows_cleared = self.remove_full_rows()
                    # Increase score for cleared rows
                    self.score += self.calculate_game_score(rows_cleared)
                    self.current_piece = self.new_piece()

                    # Check for game over
                    if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                        self.game_over = True

                fall_time = 0  # Reset fall time

            # Draw the game state
            self.draw()


if __name__ == "__main__":
    import tetris_game_config as game_configuration
    game = Tetris(game_configuration)
    game.run()
