import pybullet_envs
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
import gym_cartpole_swingup

if __name__ == '__main__':
    env_id = 'CartPoleSwingUp-v0'
    env = gym.make(env_id)
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2,
                  env_id=env_id, input_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=256, layer_1_size=256,
                  layer_2_size=256, n_actions=env.action_space.shape[0])

    n_games = 800
    filename = env_id + '_' + str(n_games) + 'games_scale' + '_' + \
        str(agent.scale) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_
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

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
