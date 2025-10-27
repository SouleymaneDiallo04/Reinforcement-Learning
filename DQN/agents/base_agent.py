from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
    @abstractmethod
    def act(self, state):
        pass
    
    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def replay(self, batch_size):
        pass
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")