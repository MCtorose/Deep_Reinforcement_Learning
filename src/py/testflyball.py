import os
import sys
os.add_dll_directory(os.environ['mingwPath'])
pybind_path = os.path.join(r"E:\\Desktop\\drl\\build")
sys.path.insert(0, pybind_path)

import flyBall as env

# 初始化环境
env.reset()

# 初始化状态量
states=[]
actions=[]
rewards=[]


episodes=30
for i in range(episodes):
    states=[]
    actions=[]
    rewards=[]      
    env.reset()
    while not env.done():
        state = env.state()
        action = 0
        reward = env.reward()
        env.step(action)
        

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        print(episodes,state,action,reward)


    pass

print("over")





