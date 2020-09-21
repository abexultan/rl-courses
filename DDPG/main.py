import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve


if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=1e-4, beta=1e-3,
                  input_dims=env.observation_space.shape,
                  tau=1e-3, batch_size=64, fc1_dims=400,
                  fc2_dims=300, n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' +\
        str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        score = 0
        done = False
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        avg = np.mean(score_history[-100:])

        if avg > best_score:
            best_score = avg
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
