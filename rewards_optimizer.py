import random
import numpy as np
from reward_center import RewardCenter
from environment_tetris import TetrisEnv
import random
import sys
from agent_bfo import BruteForceAgent

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


def convert_to_rewards_dict(combo):
    reward_combo = {}
    counter = 0
    for i, (key, val) in enumerate(rewards_ranges.items()):
        if isinstance(val, dict):
            tmp = {}
            for i2, (key2, val2) in enumerate(val.items()):
                tmp[key2] = combo[counter]
                counter += 1
            reward_combo[key] = (True, tmp)
        else:
            reward_combo[key] = (True, combo[counter])
            counter += 1

    return reward_combo


def get_combinations(arrays):
    if not arrays:
        return [[]]
    
    result = []
    for item in arrays[0]:
        for combination in get_combinations(arrays[1:]):
            result.append([item] + combination)
    
    return result


def fetch_lists_from_dict(rr: dict):
    list_of_arrays = []
    for _, (key, val) in enumerate(rr.items()):
        if isinstance(val, dict):
            tmp_list = fetch_lists_from_dict(val)
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


runner_args = {
    'random_seed':          26392639,
    'score_cutoff'     :    400000,
    'mode'             :    'bfo', # 'user', 'random_watch', 'q_learning', 'dqn', 'bfo'
    'print_reward_calc':    False,
    'publish_rewards'  :    False,
    'debug_grid'       :    False,
    'render'           :    False,
    'render_pause_sec' :    0.0,
    }


# Main function for rewards optimizer execution.
if __name__ == "__main__":
    """
    """
    total_len = print_rewards_ranges(rewards_ranges)
    list_of_vals = fetch_lists_from_dict(rewards_ranges)
    combinations = get_combinations(list_of_vals)
    print(f'Total number of combinations: {len(combinations)}')
    print(f'Total number of runs: {len(combinations)*num_replicates}')

    random.seed(runner_args['random_seed'])
    random_seeds = [random.randint(0, sys.maxsize) for x in range(num_replicates)]
    highest_combo = 0
    highest_indx = -1
    for num, combo in enumerate(combinations):
        combo_dict = convert_to_rewards_dict(combo)
        combo_score = 0
        for random_seed in random_seeds:
            runner_args['random_seed'] = random_seed
            env = TetrisEnv(runner_args, combo_dict)
            agent = BruteForceAgent(env, False, False)
            done = False
            while not done:
                done, _, _ = agent.step()
            combo_score += env.score
        avg_score = combo_score/num_replicates
        print(f'Combo {num}/{len(combinations)}: Average Score = {avg_score:.2f}')
        if (avg_score > highest_combo):
            highest_combo = avg_score
            highest_indx = num
    
    print(f'Best rewards combo had average score of {highest_combo}:\n{convert_to_rewards_dict(combinations[highest_indx])}')