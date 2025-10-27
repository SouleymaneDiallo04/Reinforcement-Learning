import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_training_results(self, scores_simple, scores_replay, save_path=None):
        plt.figure(figsize=(12, 5))
        
        # Raw scores
        plt.subplot(1, 2, 1)
        plt.plot(scores_simple, alpha=0.7, label='Sans Replay')
        plt.plot(scores_replay, alpha=0.7, label='Avec Replay')
        plt.title('Scores par épisode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Moving averages
        plt.subplot(1, 2, 2)
        window = 50
        simple_smooth = [np.mean(scores_simple[i:i+window]) for i in range(len(scores_simple)-window)]
        replay_smooth = [np.mean(scores_replay[i:i+window]) for i in range(len(scores_replay)-window)]
        
        plt.plot(simple_smooth, label='Sans Replay (moyenne)')
        plt.plot(replay_smooth, label='Avec Replay (moyenne)')
        plt.title('Scores (moyenne mobile)')
        plt.xlabel('Episode')
        plt.ylabel('Score moyen')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_q_values(self, q_table, save_path=None):
        """Plot Q-values heatmap for a fixed goal position"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        goal_positions = [(4, 4), (0, 4), (4, 0), (2, 2)]
        
        for idx, (gx, gy) in enumerate(goal_positions):
            q_grid = np.zeros((5, 5))
            for x in range(5):
                for y in range(5):
                    q_grid[y, x] = np.max(q_table.get((x, y, gx, gy), [0]))
            
            im = axes[idx].imshow(q_grid, cmap='viridis')
            axes[idx].set_title(f'Q-values max - Objectif ({gx}, {gy})')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()