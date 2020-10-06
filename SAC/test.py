import gym
import time
import gym_cartpole_swingup

if __name__ == "__main__":
    env = gym.make("CartPoleSwingUp-v0")
    time.sleep(3)
    for i in range(40000):
        step = 0
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            obs_, reward, done, _ = env.step(action)
            step += 1
            print(f"Step {step}, done flag {done}, reward {reward}")
            time.sleep(.01)