import gym
from networkagent import PolicyGradientAgent
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):

    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.label('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(lr=5e-4, gamma=0.99, input_dims=[8],
                                n_actions=4)
    fname = 'REINFORCE_' + 'lunar_lr' + str(agent.lr) + '_' + str(n_games)\
            + 'games'
    figure_file = './plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = observation_

        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)
