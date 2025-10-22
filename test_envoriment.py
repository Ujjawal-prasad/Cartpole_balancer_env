from gymnasium.envs.registration import register
register(id='CartPoleGazebo-v0', entry_point='refined_envoriment:CartPoleGazebo',)
import gymnasium as gym
env = gym.make('CartPoleGazebo-v0')
obs, info = env.reset()
print (f"starting_observation:{obs}")
episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    obs , reward, terminated, truncated, info = env.step(action)
    total_reward+=reward
    episode_over = terminated or truncated

print(f"episode finished! Total reward: {total_reward}")
env.close()