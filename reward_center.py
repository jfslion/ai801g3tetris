# Imports.
import time
import numpy as np

class RewardCenter:
    """"""
    def __init__(self, reward_params, print_calc = True, publish_rewards = False):
        self.print_reward_calc = print_calc
        self.publish_rewards = publish_rewards
        self.grid = []
        self.prev_grid = []
        self.setup = reward_params
        self.types = ['lines_cleared',
                        'max_height',
                        'max_height_diff',
                        'cells_blocked',
                        'bumpiness',
                        'total_pieces',
                        'bad_movement',
                        'unoccupied_edges',]

    def calc_reward(self, *args):
        """
        Main function for calculating total reward of a step.
        args map
        [0] previous grid
        [1] uncleared_grid
        [2] new_grid
        [3] num_lines_cleared
        [4] total_pieces
        [5] bad_movement_bool
        """
        # Unpack args.
        self.prev_grid = args[0]
        self.uncleared_grid = args[1]
        self.grid = args[2]
        num_lines_cleared = args[3]
        total_pieces = args[4]
        bad_movement_bool = args[5]
        # Initialize the reward.
        reward = 0
        reward_meta = {}

        for key in self.setup:

            # Add a quadratic reward for clearing lines to incentivize combos.
            if key == "lines_cleared" and self.setup[key][0]:
                lines_cleared = self.setup[key][1]['mult'] * (num_lines_cleared ** 2.0)
                reward_meta[key] = lines_cleared
                reward += lines_cleared
                if self.print_reward_calc:
                    print(f'Num lines cleared: {num_lines_cleared}')

            # Encourage keeping the max column height low.
            if key == "max_height" and self.setup[key][0]:
                max_height = self.calc_max_height() * self.setup[key][1]['mult']
                reward_meta[key] = max_height
                reward += max_height

            # Encourage keeping the max column height differetial low.
            if key == "max_height_diff" and self.setup[key][0]:
                max_height_diff = self.calc_max_height_diff() * self.setup[key][1]['mult']
                reward_meta[key] = max_height_diff
                reward += max_height_diff

            # Count the number of spaces the agent can't fill
            if key == "cells_blocked" and self.setup[key][0]:
                cells_blocked = self.calculate_unreachable_spaces() * self.setup[key][1]['mult']
                reward_meta[key] = cells_blocked
                reward += cells_blocked

            # Minimize the height difference across rows.
            if key == "bumpiness" and self.setup[key][0]:
                prev_bumpiness = self.calculate_bumpiness(self.prev_grid)
                new_bumpiness = self.calculate_bumpiness(self.grid)
                bumpiness = 0
                if (new_bumpiness - prev_bumpiness) > 0:
                    bumpiness = (new_bumpiness - prev_bumpiness) * self.setup[key][1]['mult']
                if self.print_reward_calc:
                    print(f'Num bumps added: {bumpiness}')
                reward_meta[key] = bumpiness
                reward += bumpiness

            # Add a reward for the total number of peices placed.
            if key == "total_pieces" and self.setup[key][0]:
                reward_meta[key] = total_pieces * self.setup[key][1]['mult']
                reward += total_pieces * self.setup[key][1]['mult']

            # Highly penalize unnecesary movements.
            if key == "bad_movement" and self.setup[key][0]:
                bad_movement_hit = 0
                if bad_movement_bool:
                    bad_movement_hit = self.setup[key][1]['const']
                reward_meta[key] = bad_movement_hit
                reward += bad_movement_hit

            # Minimize unoccupied edges.
            if key == "unoccupied_edges" and self.setup[key][0]:
                unoccupied_edges_val = 0
                unoccupied_edges_set = self.calc_edge_reward()
                if len(unoccupied_edges_set) > 0:
                    unoccupied_edges_val = len(unoccupied_edges_set) * self.setup[key][1]['mult']
                reward_meta['unoccupied_edges'] = unoccupied_edges_val
                reward += unoccupied_edges_val

        reward_meta['total'] = reward
        reward = np.float32(reward)


        return reward, reward_meta


    def count_new_blocks(self):
        """
        A reward that gives points for each new peice in a row.
        The reward increases as number of peices in a row increase.
        """
        new_blocks_count = []
        total_blocks_count = []
        cumulative_sums = []

        for prev_row, new_row in zip(self.prev_grid, self.grid):
            # Calculate the change in the row
            change = [new - prev for new, prev in zip(new_row, prev_row)]

            # Count the new blocks (number of 1s in the change)
            new_count = change.count(1)

            # Calculate the total number of blocks in the new row
            total_count = sum(new_row)

            # Calculate the cumulative sum based on total_count and new_count
            if new_count > 0:
                cumulative_sum = sum(
                    range(total_count, total_count - new_count, -1))
            else:
                cumulative_sum = 0

            new_blocks_count.append(new_count)
            total_blocks_count.append(total_count)
            cumulative_sums.append(cumulative_sum)

        if self.print_reward_calc:
            print(f'Total Blocks in each row: {total_blocks_count}')
            print(f'New Blocks in each row: {new_blocks_count}')
            print(f'Cum Sums: {cumulative_sums}')

        time.sleep(1)
        return new_blocks_count, total_blocks_count, cumulative_sums


    def calc_average_height(self):
        """
        Returns the average height of all of the columns of the tetris board.
        """
        width = len(self.grid[0])
        heights = [0] * width

        for col in range(width):
            for row in range(len(self.grid)):
                if self.grid[row][col] == 1:
                    heights[col] = len(self.grid) - row
                    break

        if self.print_reward_calc:
            print(f'Column Heights: {heights}')
            print(f'Average Height: {sum(heights) / width}')

        return sum(heights) / width


    def calc_edge_reward(self):
        """
        Determine the number of open edges on the most recenly placed peice.
        Do not count edges twice and consider the edge of the board as an occupied space.  
        """
        rows = len(self.grid)
        cols = len(self.grid[0])
        unoccupied_spaces = set()

        def count_unoccupied_edges(row, col):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    if self.grid[new_row][new_col] == 0:
                        unoccupied_spaces.add((new_row, new_col))
                else:
                    # Count edges of the board as occupied.
                    pass

        for row in range(rows):
            for col in range(cols):
                if self.grid[row][col] == 1 and self.prev_grid[row][col] == 0:
                    # This is a newly placed piece.
                    count_unoccupied_edges(row, col)

        if self.print_reward_calc:
            print(f'Unoccupied edges: {len(unoccupied_spaces)}')

        return unoccupied_spaces


    def calculate_unreachable_spaces(self):
        """
        The agent can only send pieces straight down.
        Assign a penalty for the number of spaces an agent will not be able to reach.
        """
        height = len(self.grid)
        width = len(self.grid[0])
        unreachable_count = 0
        prev_unreachable_count = 0

        for col in range(width):
            found_piece = False
            for row in range(height):
                if self.grid[row][col] == 1:
                    found_piece = True
                elif found_piece and self.grid[row][col] == 0:
                    unreachable_count += 1
        
        for col in range(width):
            found_piece = False
            for row in range(height):
                if self.prev_grid[row][col] == 1:
                    found_piece = True
                elif found_piece and self.prev_grid[row][col] == 0:
                    prev_unreachable_count += 1

        if self.print_reward_calc:
            print(f'Number of unreachable spaces: {unreachable_count - prev_unreachable_count}')

        return unreachable_count - prev_unreachable_count


    def calc_max_height(self):
        """
        # Extract the height of the tallest column.
        """
        width = len(self.grid[0])
        heights = [0] * width
        x=2
        for col in range(width):
            for row in range(len(self.grid)):
                if self.grid[row][col] == 1:
                    heights[col] = len(self.grid) - row
                    break

        if self.print_reward_calc:
            print(f'Max Height: {max(heights)}')

        return max(heights)


    def calc_max_height_diff(self):
        """
        # Extract the height of the tallest column.
        """
        width = len(self.grid[0])
        heights = [0] * width

        for col in range(width):
            for row in range(len(self.grid)):
                if self.grid[row][col] == 1:
                    heights[col] = len(self.grid) - row
                    break

        if self.print_reward_calc:
            print(f'Max Height Diff: {max(heights) - min(heights)}')

        return max(heights) - min(heights)


    def calculate_bumpiness(self, aGrid):
        """
        Sum up the difference in each of the heighs of the
        columns and the column next to it.
        """
        width = len(aGrid[0])
        heights = [0] * width

        # Calculate the height of each column.
        for col in range(width):
            for row in range(len(aGrid)):
                if aGrid[row][col] != 0:
                    heights[col] = len(aGrid) - row
                    break

        # Calculate the sum of differences between adjacent columns.
        bumpiness = 0
        for i in range(width - 1):
            bumpiness += abs(heights[i] - heights[i+1])

        return bumpiness