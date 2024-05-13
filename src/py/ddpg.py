import random

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import torch.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义Actor网络
# 输入状态，输出动作
# Linear layers with ReLU activation ending with tanh activation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x


# 定义Critic网络
# linear layers with ReLU activation ending with no activation
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# replay buffer
# sample,push,__len__
# deque with 15000 max length
class ReplayBuffer:
    def __init__(self, max_size=15000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(a=len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# dddg agent算法
# get_action ,update, __init__,memory
class DDPGAgent:

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3, buffer_size=15000, batch_size=64):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        return action

    def update(self, BATCH_SIZE=64):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        next_action = self.target_actor(next_state)
        target_Q = self.target_critic(next_state, next_action)
        target_Q = reward + (1 - done) * self.gamma * target_Q
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    pass
