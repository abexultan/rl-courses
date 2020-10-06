import gym
import gym_cartpole_swingup

if __name__ == "__main__":

    env = gym.make("CartPoleSwingUp-v0")
    n_games = 250

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            env.render()
            observation = observation_
            score += reward
        print('episode ', i, 'score %.1f' %score)