import gym

amb = gym.make("LunarLander")

amb.reset()

for step in range(200):
  amb.render()
  obs, recompensa, terminado, info = amb.step(amb.action_space.sample())
  print(recompensa)

amb.close()