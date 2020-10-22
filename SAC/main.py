import numpy as np
from agent import Agent
import time
from cartpole import CartPoleEnvFullStateSwingUp as myEnv
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass values")
    parser.add_argument("--tau", type=float, help="Need a floating number")
    parser.add_argument("--theta_threshold", type=float,
                        help="Need a floating number")
    parser.add_argument("--k1", type=float)
    parser.add_argument("--k2", type=float)
    parser.add_argument("--k3", type=float)
    parser.add_argument("--k4", type=float)
    args = parser.parse_args()
    delta_t = args.tau
    theta_threshold = args.theta_threshold
    k1 = args.k1
    k2 = args.k2
    k3 = args.k3
    k4 = args.k4

    env_id = 'SwingUp_deltat_' + str(delta_t) + '_theta_threshold_'\
        + str(theta_threshold) + '_k1_' + str(k1) + '_k2_' + str(k2)\
        + '_k3_' + str(k3) + '_k4_' + str(k4)

    load_checkpoint = False
    env = myEnv(mode='test' if load_checkpoint else 'train',
                k1=k1, k2=k2, k3=k3, k4=k4)
    env.tau = delta_t
    env.theta_threshold_radians = theta_threshold

    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2,
                  env_id=env_id, input_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=256, layer_1_size=256,
                  layer_2_size=256, n_actions=env.action_space.shape[0])

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        env.pos_desired = 1.5
        agent.load_models()
        angle = []
        position = []
        thetas = []

    n_games = 1 if load_checkpoint else 800
    max_steps = 1000 if load_checkpoint else 2000 * int(0.01 // delta_t)

    filename = env_id + '_' + str(n_games) + 'games_scale' + '_' + \
        str(agent.scale) + '.png'
    figure_file = 'plots/' + filename

    for i in range(n_games):
        steps = 0
        score = 0
        done = False
        observation = env.reset()
        while not done:
            if load_checkpoint:
                angle.append(env.theta_desired - observation[2])
                position.append(env.pos_desired - observation[0])
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            if steps == max_steps:
                done = True
                reward = 0

            score += reward

            if not load_checkpoint:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
            else:
                env.render()
                time.sleep(0.01)

            observation = observation_

        if load_checkpoint:
            env.close()
        score_history.append(score)
        avg = np.mean(score_history[-100:])

        if avg > best_score:
            best_score = avg
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg,
              'steps %d' % steps, env_id,
              'scale', agent.scale)

    if load_checkpoint:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Delta_t = {delta_t}, threshold = {theta_threshold}' +
                     f' k1 = {k1}, k2 = {k2}, k3 = {k3}, k4 = {k4}')
        ax1.plot(angle, label='pole angle')
        ax1.plot(np.zeros((len(angle), 1)), 'r--', label='0 radian')

        ax1.set_xlabel('episodes')
        ax1.set_ylabel('pole angle')
        ax1.legend()

        ax2.plot(position, label='cart position')
        ax2.plot(np.zeros((len(position), 1)) + env.pos_desired, 'r--',
                 label='reference position')

        ax2.set_xlabel('episodes')
        ax2.set_ylabel('cart position')
        ax2.legend()
        plt.savefig(env_id + '_pos_angle.png')
    env.close()
