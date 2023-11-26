from Agent import DQNAgent
from collections import deque
import numpy as np
import torch
import random
import itertools
import gym

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

env_name = 'CartPole-v0'
env = gym.make(env_name)

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

agent = DQNAgent(env, EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA)

obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _ = env.step(action)
    Transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(Transition)
    obs = new_obs

    if done:
        obs = env.reset()

obs = env.reset()

for step in itertools.count():
    
    action = agent.choose_action(step, obs)

    new_obs, rew, done, _ = env.step(action)
    Transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(Transition)
    obs = new_obs

    episode_reward += rew
    
    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    agent.calculate_target(new_obses_t, rews_t, dones_t)
    agent.calculate_action_qvalues(obses_t, actions_t)
    agent.optimize_network()

    if step % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    if step % 1000 == 0:
        print()
        print()
        print("step ", step)
        print("Avg reward ", np.mean(rew_buffer))