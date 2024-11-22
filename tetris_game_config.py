# Define colors using RGB tuples
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)  # Color for the border

# Game dimensions
BLOCK_SIZE = 30  # Size of each tetromino block in pixels
GRID_WIDTH = 10  # Number of columns in the game grid
GRID_HEIGHT = 20  # Number of rows in the game grid
BORDER_WIDTH = 4  # Width of the border around the game area in pixels
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + BORDER_WIDTH * \
    2 + 200  # Total screen width, including space for score
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT + BORDER_WIDTH  # Total screen height

# Define tetromino shapes using 2D lists
# Each sublist represents a row, and 1 indicates a filled block
SHAPES = [
    [[1, 1, 1, 1]],  # I-shape
    [[1, 1], [1, 1]],  # O-shape
    [[1, 1, 1], [0, 1, 0]],  # T-shape
    [[1, 1, 1], [1, 0, 0]],  # L-shape
    [[1, 1, 1], [0, 0, 1]],  # J-shape
    [[1, 1, 0], [0, 1, 1]],  # S-shape
    [[0, 1, 1], [1, 1, 0]]  # Z-shape
]

# Colors for each tetromino shape
COLORS = [CYAN, YELLOW, MAGENTA, RED, GREEN, BLUE, ORANGE]