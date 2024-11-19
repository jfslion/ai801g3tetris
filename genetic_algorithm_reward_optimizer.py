import json
import time
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from tetris_without_pygame import TetrisEnvNoGame
from agent_mdp import BruteForceAgent

def mdp_agent_optimization(input_args):
    
        # Configuration Items
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

    # Rewards configuration items.
    rewards_config = {
        'lines_cleared': (True, {'mult': input_args[0], 'exp': input_args[1]}),
        'max_height': (True, {'mult': input_args[2]}),
        'max_height_diff': (True, {'mult': input_args[3]}),
        'cells_blocked': (True, {'mult': input_args[4]}),
        'bumpiness': (True, {'mult': input_args[5]}),
        'total_pieces': (False, {'mult': input_args[6]}),
        'bad_movement': (True, {'const': input_args[7]}),
        'unoccupied_edges': (True, {'mult': input_args[8], 'scale': input_args[9]}),
    }

    env = TetrisEnvNoGame(runner_args, rewards_config)

    # Setup agent played mode.
    done = False
    state, reward = env.reset()

    agent = BruteForceAgent(env, False, False)
    while not done:
        done, reward_meta, action_meta = agent.step()

    return env.score

def f(X):
    return -mdp_agent_optimization(X)  # Negative because we want to maximize

varbound = np.array([
    [0, 200],  # lines_cleared mult
    [1, 3],    # lines_cleared exp
    [-10, 10],  # max_height mult
    [-10, 10],  # max_height_diff mult
    [-10, 10],  # cells_blocked mult
    [-10, 10],  # bumpiness mult
    [0, 0],    # total_pieces mult
    [-200, -199], # bad_movement const
    [-10, 10],   # unoccupied_edges mult
    [-10, 20]    # unoccupied_edges scale
])

algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 50,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}
# Example input arguments
sample_input_args = [
    100.0,  # lines_cleared mult
    2.0,    # lines_cleared exp
    -2.0,   # max_height mult
    -2.0,   # max_height_diff mult
    -10.0,  # cells_blocked mult
    -1.0,   # bumpiness mult
    1.5,    # total_pieces mult
    -100.0, # bad_movement const
    1.0,    # unoccupied_edges mult
    10.0    # unoccupied_edges scale
]

# Call the function with the sample input
result = mdp_agent_optimization(sample_input_args)
print(f"Optimization result: {result}")

start_time = time.time()


model = ga(function=f,
           dimension=10,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

output_dict = model.run()
report = model.report

end_time = time.time()
execution_time = end_time - start_time

print("\nGenetic Algorithm Optimization Results:")
print("=======================================")
print(f"Best solution (weights): {output_dict['variable']}")
print(f"Best fitness score: {-output_dict['function']}")  # Negate if maximizing
print(f"Number of generations: {report['generation']}")
print(f"Execution time: {execution_time:.2f} seconds")

print("\nAlgorithm Parameters:")
for key, value in algorithm_param.items():
    print(f"{key}: {value}")

print("\nEvolution of best solution:")
for generation, score in enumerate(report['function']):
    if generation % 10 == 0:  # Print every 10th generation to keep output manageable
        print(f"Generation {generation}: Best Score = {-score}")  # Negate if maximizing

# Save results to a file
results = {
    'best_weights': output_dict['variable'].tolist(),
    'best_score': -output_dict['function'],
    'generations': report['generation'],
    'execution_time': execution_time,
    'algorithm_parameters': algorithm_param,
    'evolution': [-score for score in report['function']]  # Negate if maximizing
}

with open('ga_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nDetailed results saved to 'ga_results.json'")