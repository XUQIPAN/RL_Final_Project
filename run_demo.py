import gymnasium as gym
import read_the_boss  # registers ReadTheBoss-v0

env = gym.make("ReadTheBoss-v0", opponent="balanced", render_mode="ansi")
obs, info = env.reset(seed=7)
action = env.action_space.sample(mask=info["action_mask"])
obs, reward, terminated, truncated, info = env.step(action)
print(env.render())
env.close()
