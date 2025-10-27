import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from .base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, use_replay=True, model_path="models/dqn_agent.pth"):
        super().__init__(state_size, action_size)
        
        self.use_replay = use_replay
        self.model_path = model_path
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Model and optimizer
        from models.networks import DQN
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create model directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def remember(self, state, action, reward, next_state, done):
        if self.use_replay:
            self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size):
        if not self.use_replay or len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = next_state.unsqueeze(0)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()
            
            state = state.unsqueeze(0)
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
    
    def learn(self, state, action, reward, next_state, done):
        """Direct learning without replay"""
        target = reward
        if not done:
            next_state = next_state.unsqueeze(0)
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
        
        state = state.unsqueeze(0)
        target_f = self.model(state).detach().clone()
        target_f[0][action] = target
        
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(self.model(state), target_f)
        loss.backward()
        self.optimizer.step()
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.model_path)
    
    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            return True
        return False