from env import SimpleEnv

env = SimpleEnv(display=True)

env.reset()

for _ in range(400):
    action = env.action_space.sample()
    new_obs, rew, done, _ = env.step(action)
    print(rew)