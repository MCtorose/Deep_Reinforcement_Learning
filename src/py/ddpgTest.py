import gym
import torch
import numpy as np
from ddpg import Actor
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Pendulum-v1', render_mode='rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

MAX_STEPS = 100

# 加载模型
actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(r'E:\Desktop\drl\model_save\actor.pth'))

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Pendulum-v1')
clock = pygame.time.Clock()

for episode in range(10):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(MAX_STEPS):
        # 渲染环境
        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2))
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (800, 600))
        screen.blit(frame, (0, 0))
        pygame.display.flip()

        clock.tick(60)
        # 选择动作
        state = torch.FloatTensor(state).to(device)
        action = actor(state).cpu().detach().numpy()
        # 执行动作
        next_state, reward, done, _, _ = env.step(2 * action)
        episode_reward += reward
        # 更新状态
        state = next_state
        print(f"Episode {episode}, Step {step}, Reward {reward}")
        if done:
            break

    # 关闭环境
env.close()
