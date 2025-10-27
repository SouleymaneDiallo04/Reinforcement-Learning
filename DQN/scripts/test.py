import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.gridworld import GridWorld
from agents.dqn_agent import DQNAgent

def test_agent(agent_path, num_episodes=10, agent_name="DQN Agent"):
    """Test a trained agent"""
    
    env = GridWorld()
    agent = DQNAgent(state_size=4, action_size=4)
    
    if not agent.load_checkpoint():
        print(f" Could not load model from {agent_path}")
        return
    
    print(f"\n Testing {agent_name}")
    print("=" * 40)
    
    success_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            if done:
                success_count += 1
                total_steps += steps
                print(f" Episode {episode}: Success in {steps} steps "
                      f"(Agent: {env.agent_pos}, Goal: {env.goal_pos})")
                break
                
            if steps >= 50:
                print(f" Episode {episode}: Failed after 50 steps "
                      f"(Agent: {env.agent_pos}, Goal: {env.goal_pos})")
                break
                
            steps += 1
            state = next_state
    
    success_rate = success_count / num_episodes * 100
    avg_steps = total_steps / success_count if success_count > 0 else float('inf')
    
    print(f"\n📊 Final Results for {agent_name}:")
    print(f"   Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"   Average Steps (on success): {avg_steps:.1f}")

def main():
    print(" GridWorld DQN Testing")
    
    # Test both agents
    test_agent("models/dqn_simple.pth", agent_name="DQN Simple")
    test_agent("models/dqn_replay.pth", agent_name="DQN with Replay")

if __name__ == "__main__":
    main()