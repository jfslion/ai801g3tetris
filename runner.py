import time
from environment_tetris import TetrisEnv
from agent_q_learning import QLearningAgent


# Configuration Items
runner_args = {
    'random_seed':          26392639,
    'score_cutoff'     :    400000,
    'mode'             :    'user', # 'user', 'q_learning', 'dqn'
    'print_reward_calc':    True,
    'publish_rewards'  :    True,
    'debug_grid'       :    True,
    'render'           :    True,
    }

# Rewards configuration items.
rewards_config = {
    'lines_cleared'     : (True, {'mult': 100.0, 'exp': 2.0}),
    'max_height'        : (True, {'mult': -2.0}),
    'cells_blocked'     : (True, {'mult': -1.0}),
    'bumpiness'         : (True, {'mult': -1.0}),
    'total_pieces'      : (True, {'mult': 1.5}),
    'bad_movement'      : (True, {'const': -100.0}),
    'unoccupied_edges'  : (True, {'mult': 1.0, 'scale': 10.0}),
}


# Main function for runner execution.
if __name__ == "__main__":
    """"""
    run_mode = runner_args['mode']
    env = TetrisEnv(runner_args, rewards_config)

    # Execute user specified run_mode.
    match run_mode:

        case 'user':
            env.run()

        case 'q_learning':            
            state_size = env.get_state_size()
            action_space = env.action_space
            agent = QLearningAgent(state_size, action_space)
            done = False
            state, reward = env.reset()
            # env.run_woth_reward()
            while not done:
                action = np.random.randint(0, 43)
                observation, reward, done, _, _ = env.step(action)
                env.render()
                time.sleep(.5)  # Pauses for 1 second
                print(f"Action: {action}, Reward: {reward}, Done: {done}")

        case 'dqt':
            #TODO
            pass
            # RENDER = True
            # num_episodes = 1000
            # for episode in range(num_episodes):
            #     state = env.reset()
            #     state = agent.state_to_key(state)
            #     total_reward = 0
            #     done = False

            #     while not done:
            #         action = agent.choose_action(state)
            #         next_state, reward, done, truncated, info = env.step(action)
            #         if RENDER:
            #             env.render()
            #         next_state = agent.state_to_key(next_state)

            #         agent.learn(state, action, reward, next_state, done)

            #         state = next_state
            #         total_reward += reward

            #     print(f"Episode {
            #           episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")