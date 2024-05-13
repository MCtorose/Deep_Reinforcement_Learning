# DDPG
import torch
import gym
from ddpg import DDPGAgent
import numpy as np

env = gym.make(id='Pendulum-v1')

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = DDPGAgent(STATE_DIM, ACTION_DIM)

MAX_EPISODES = 200
MAX_EP_STEPS = 200
BATCH_SIZE = 64
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    ep_reward = 0
    for step in range(MAX_EP_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM)
        next_state, reward, done, _, _ = env.step(2 * action)
        agent.buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        if len(agent.buffer) >= BATCH_SIZE:
            agent.update(BATCH_SIZE)
        if done:
            break
    print("Episode: {}, Reward: {}".format(episode, ep_reward))

env.close()
torch.save(agent.actor.state_dict(), 'actor.pth')
torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
