import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

vec_env = model.get_env()
obs = vec_env.reset()

# Modelo PPO
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()