import numpy as np
from agent import Agent
import time
from cartpole_swingup import CartPoleSwingUpV0
# import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass values")
    parser.add_argument("--tau", type=float, help="Need a floating number")
    args = parser.parse_args()
    delta_t = args.tau
    env_id = 'SwingUp_deltat_' + str(delta_t)

    load_checkpoint = False
    env = CartPoleSwingUpV0()
    env.tau = delta_t

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
    max_steps = 500 if load_checkpoint else 1000 * int(0.01 // delta_t)

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

    env.close()
