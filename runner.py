import time
import numpy as np
from env_tetris import TetrisEnv
from agent_q_learning import QLearningAgent
import matplotlib.pyplot as plt
from agent_mdp import AgentMDP

def plot_cumulative_rewards(reward_history):
    steps = range(len(reward_history))
    
    plt.figure(figsize=(12, 8))
    
    for reward_type in reward_history[0].keys():
        rewards = [step[reward_type] for step in reward_history]
        cumulative_rewards = np.cumsum(rewards)
        plt.plot(steps, cumulative_rewards, label=f'{reward_type}')
    
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()

def plot_rewards(reward_history):
    steps = range(len(reward_history))
    
    plt.figure(figsize=(12, 8))
    
    for reward_type in reward_history[0].keys():
        rewards = [step[reward_type] for step in reward_history]
        plt.plot(steps, rewards, label=reward_type)
    
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Rewards Per Step')
    plt.legend()
    plt.grid(True)
    # plt.show()

# Configuration Items
runner_args = {
    'random_seed':          26392639,
    'score_cutoff'     :    400000,
    'mode'             :    'mdp', # 'user', 'random_watch', 'q_learning', 'dqn', 'bfo'
    'print_reward_calc':    False,
    'publish_rewards'  :    True,
    'debug_grid'       :    False,
    'render'           :    True,
    'render_pause_sec' :    3.0,
    }

# Rewards configuration items.
rewards_config = {
    'lines_cleared'     : (True, {'mult': 100.0}),
    'max_height'        : (False, {'mult': -1.0}),
    'max_height_diff'   : (True, {'mult': -8.0}),
    'cells_blocked'     : (True, {'mult': -100.0}),
    'bumpiness'         : (True, {'mult': -20}),
    'total_pieces'      : (False, {'mult': 1.0}),
    'bad_movement'      : (False, {'const': -100.0}),
    'unoccupied_edges'  : (True, {'mult': -10.0}),
}

# Main function for runner execution.
if __name__ == "__main__":
    """
    """
    run_mode = runner_args['mode']
    env = TetrisEnv(runner_args, rewards_config)

    # Short the runner if it is a user game.
    if run_mode == 'user':
        env.run()
        exit

    # Setup agent played mode.
    done = False
    state, reward = env.reset()
    print_reward = runner_args['print_reward_calc']
    render = runner_args['render']
    render_pause = runner_args['render_pause_sec']

    # Execute user specified run_mode.
    match run_mode:

        case 'random_watch':
            """
            Useful for debugging.
            """
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, reward_meta = env.step(action)
                if render:
                    env.draw(reward_meta)
                    time.sleep(render_pause)
                if print_reward:
                    print(f"Action: {action}, Reward: {reward}, Done: {done}")

        case 'q_learning':
            """
            """
            agent = QLearningAgent(env.get_state_size(), env.action_space)
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, reward_meta = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                if render:
                    env.draw(reward_meta)
                    time.sleep(render_pause)
                if print_reward:
                    print(f"Action: {action}, Reward: {reward}, Done: {done}")

        case 'mdp':
            """
            """
            reward_history = []
            agent = AgentMDP(env, False, False)
            while not done:
                done, reward_meta, action_meta = agent.step()
                reward_history.append(reward_meta.copy())

            # plot the rewards from this episode
            env.draw(reward_meta, action_meta, 1)
            plot_rewards(reward_history)
            plot_cumulative_rewards(reward_history)
            plt.show()

        case 'dqt':
            """
            """
            pass

    print(f'Final Score: {env.score}')
            # #TODO
            # num_episodes = 1000
            # for episode in range(num_episodes):
            #     state = env.reset()
            #     state = agent.state_to_key(state)
            #     total_reward = 0
            #     done = False

            #     while not done:
            #         action = agent.choose_action(state)
            #         next_state, reward, done, reward_meta = env.step(action)
            #         if render:
            #             env.draw(reward_meta)
            #         next_state = agent.state_to_key(next_state)

            #         agent.learn(state, action, reward, next_state, done)

            #         state = next_state
            #         total_reward += reward

            #     print(f"Episode {
            #           episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")