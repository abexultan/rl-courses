import numpy as np
from agent import Agent
import time
from cartpole import CartPoleEnv, CartPoleEnvState
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env_id = 'CustomCurtpoleEnv_exp_reward_state'
    env = CartPoleEnvState(mode='train')
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
    thetas = []

    if load_checkpoint:
        agent.load_models()
    
    angle = []
    
    max_steps = 500 if load_checkpoint else 2000

    for i in range(n_games):  
        steps = 0
        score = 0
        done = False
        observation = env.reset()
        while not done:
            angle.append(observation[2])
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
                time.sleep(0.001)

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
        plt.plot(angle, label='pole angle')
        plt.plot(np.zeros((len(angle), 1)), 'r--', label='0 radian')

        plt.xlabel('episodes')
        plt.ylabel('pole angle')
        plt.legend()
        plt.savefig("pole.png")

    env.close()
