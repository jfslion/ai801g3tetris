import random
import numpy as np
from reward_center import RewardCenter
import math

gen_files = True
execute_optimizer = True
num_replicates = 5

rewards_ranges = {
    'lines_cleared': 
        {'mult':
            [100.0, 200.0],
         'exp':
            [2.0]},
    'max_height':
        {'mult':
            [0.0, -1.0, -2.0]},
    'max_height_diff':
        {'mult':
            [0.0, -2.0, -10.0]},
    'cells_blocked':
        {'mult':
            [0.0, -5.0, -10.0]},
    'bumpiness':
        {'mult':
            [0.0, -1.0, -3.0]},
    'total_pieces':
        {'mult':
            [0.0, 1.5]},
    'bad_movement':
        {'const':
            [0.0, -100.0]},
    'unoccupied_edges':
        {'mult':
            [0.0, 1.0, 5.0], 
         'scale':
            [5.0, 10.0]},
}


def get_run_matrix(rr: dict):
    list_of_arrays = []
    for _, (key, val) in enumerate(rr.items()):
        if isinstance(val, dict):
            tmp_list = get_run_matrix(val)
            for alist in tmp_list:
                list_of_arrays.append(alist)
        else:
            list_of_arrays.append(val)
    
    return list_of_arrays


def print_rewards_ranges(rr: dict, num_tabs = 1):
    tab_str = ''
    for i in range(num_tabs-1):
        tab_str += '\t'

    total_len = 0

    for _, (key, val) in enumerate(rr.items()):
        print(f'{tab_str}{key}:')
        if isinstance(val, dict):
            total_len += print_rewards_ranges(val, num_tabs+1)
        else:
            print(f'\t{tab_str}{val}')
            total_len += len(val)
    
    print(f'Iter total_len: {total_len}')
    return total_len

# Main function for rewards optimizer execution.
if __name__ == "__main__":
    """
    """
    if gen_files:
        total_len = print_rewards_ranges(rewards_ranges)
        run_mat = get_run_matrix(rewards_ranges)
    
    print(f'Total number of runs: {math.factorial(total_len)}')
