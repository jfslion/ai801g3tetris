import random
import numpy as np
from reward_center import RewardCenter
from environment_tetris import TetrisEnv
from tetris_without_pygame import TetrisEnvNoGame
import random
import sys
from agent_mdp import BruteForceAgent
from joblib import Parallel, delayed
import os
import datetime
import pprint
import statistics

gen_files = True
execute_optimizer = True
num_replicates = 10

rewards_ranges = {
    'lines_cleared': 
        {'mult':
            [100.0, 200.0, 300.0]},
    'max_height_diff':
        {'mult':
            [-2.0, -8.0, -20.0, -40.0]},
    'cells_blocked':
        {'mult':
            [-10.0, -25.0, -50.0, -100.0, -200.0]},
    'bumpiness':
        {'mult':
            [-1.0, -3.0, -5.0, -10.0, -20.0]},
    'unoccupied_edges':
        {'mult':
            [-1.0, -3.0, -10.0, -20.0]}, 
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



def process_combo(an_indx, a_combo, seeds, out_dir):
    run_args = {
        'random_seed':          26392639,
        'score_cutoff'     :    400000,
        'mode'             :    'mdp',
        'print_reward_calc':    False,
        'publish_rewards'  :    False,
        'debug_grid'       :    False,
        'render'           :    False,
        'render_pause_sec' :    0.0,
        }
    dict_combo = convert_to_rewards_dict(a_combo)
    scores = []
    for seed in seeds:
        run_args['random_seed'] = seed
        env = TetrisEnvNoGame(run_args, dict_combo)
        agent = BruteForceAgent(env, False, False)
        done = False
        while not done:
            done, _, _ = agent.step()
        scores.append(env.score)
    
    ave_score = sum(scores)/len(seeds)
    std_dev = statistics.stdev(scores)

    score_str = [f'{a_score}\n' for a_score in scores]

    out_file = os.path.join(out_dir, f'combo_{an_indx}_{ave_score:.0f}_{std_dev:.0f}')
    with open(out_file, 'w') as f_out:
        f_out.writelines(score_str)


def publish_results(out_dir):
    combo_files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f)) and f.startswith('combo')]
    combo_scores = {}
    for combo_file in combo_files:
        combo_num = int(combo_file.split("_")[1])
        with open(os.path.join(out_dir, combo_file), 'r') as f:
            scores = []
            for line in f:
                try:
                    number = float(line.strip())
                    scores.append(number)
                except ValueError:
                    print(f"Skipping line: {line.strip()} (not a number)")
            combo_scores[combo_num] = scores
    
    # Find the highest score and highest average score.
    highest_ave = 0
    highest_ave_indx = -1
    highest_raw = 0
    highest_raw_indx = -1
    for combo_num in combo_scores:
        scores = combo_scores[combo_num]
        ave_score = sum(scores)/len(scores)
        high = max(scores)
        if (ave_score > highest_ave):
            highest_ave = ave_score
            highest_ave_indx = combo_num
        if (high > highest_raw):
            highest_raw = high
            highest_raw_indx = combo_num
    
    print(f'Highest Average Score: {highest_ave:.0f} | Combination: {highest_ave_indx}')
    print(f'Highest Game Score: {highest_raw:.0f} | Combination: {highest_raw_indx}')

# Main function for rewards optimizer execution.
if __name__ == "__main__":
    """
    """
    #total_len = print_rewards_ranges(rewards_ranges)
    # list_of_vals = fetch_lists_from_dict(rewards_ranges)
    # combinations = get_combinations(list_of_vals)
    # print(f'Total number of combinations: {len(combinations)}')
    # print(f'Total number of runs: {len(combinations)*num_replicates}')

    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    # directory_name = f'optimize_job_{timestamp}'
    # new_directory = os.path.join(working_dir, directory_name)
    # os.mkdir(new_directory)

    # summary_file = os.path.join(new_directory, 'summary_file.txt') 
    # with open(summary_file, 'w') as f:
    #     for indx, combination in enumerate(combinations):
    #         dict_combo = convert_to_rewards_dict(combination)
    #         pprint.pprint(f'Combination {indx}:', stream = f)
    #         pprint.pprint(dict_combo, stream = f)
    #         pprint.pprint(" ", stream = f)

    # random.seed(654684835)
    # seeds = [random.randint(0, sys.maxsize) for x in range(num_replicates)]

    # Parallel(n_jobs=-1)(delayed(process_combo)(indx, combo, seeds, new_directory) for indx, combo in enumerate(combinations))

    working_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(working_dir, 'optimize_job_2024-11-20_18-53-00')
    publish_results(data_dir)