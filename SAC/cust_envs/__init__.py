from gym.envs.registration import register


register(
    id = "ContCartPole-v0",
    entry_point = "cust_envs.envs:ContinuousCartPoleEnv",
    max_episode_steps = 1000,
)
