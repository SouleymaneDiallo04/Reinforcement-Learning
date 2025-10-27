class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self):
        # Environment
        self.grid_size = 5
        
        # Training
        self.episodes = 1000
        self.max_steps = 100
        self.gamma = 0.95
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory_size = 100
        
        # Model
        self.hidden_size = 64
        
        # Paths
        self.model_dir = "models"
        self.results_dir = "results"

class ExperimentConfig:
    """Configuration for different experiments"""
    
    @staticmethod
    def get_simple_config():
        return {
            'use_replay': False,
            'episodes': 500,
            'model_path': 'models/dqn_simple.pth'
        }
    
    @staticmethod
    def get_replay_config():
        return {
            'use_replay': True,
            'episodes': 500,
            'model_path': 'models/dqn_replay.pth'
        }