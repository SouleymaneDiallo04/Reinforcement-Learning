import torch
import numpy as np
import random

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        return self.get_state()
    
    def get_state(self):
        state = np.array(self.agent_pos + self.goal_pos)
        return torch.FloatTensor(state)
    
    def step(self, action):
        x, y = self.agent_pos
        
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.size-1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.size-1: x += 1
        
        self.agent_pos = [x, y]
        
        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False
            
        return self.get_state(), reward, done
    
    def get_action_meanings(self):
        return ["UP", "DOWN", "LEFT", "RIGHT"]