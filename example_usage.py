"""
OpenRA RL Environment Usage Examples
Demonstrates how to use the OpenRA environment for RL training
"""

import numpy as np
import time
from openra_env import OpenRAEnvironment, create_simple_combat_env, create_visual_env
import random

# Example 1: Basic environment interaction
def basic_example():
    """Basic example of environment usage"""
    print("🎮 Starting basic OpenRA environment example...")
    
    # Create environment
    env = create_simple_combat_env()
    
    try:
        # Reset environment
        observation, info = env.reset()
        print(f"✅ Environment reset. Initial info: {info}")
        
        # Run for a few steps with random actions
        for step in range(10):
            # Sample random action
            action = env.action_space.sample()
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step}: Reward={reward:.2f}, Units={info.get('my_unit_count', 0)}")
            
            if terminated or truncated:
                print("Episode finished!")
                break
            
            time.sleep(1)  # Slow down for demonstration
            
    except Exception as e:
        print(f"❌ Error during basic example: {e}")
    finally:
        env.close()


# Example 2: Simple AI agent
class SimpleAgent:
    """Simple rule-based agent for demonstration"""
    
    def __init__(self, env):
        self.env = env
        self.action_history = []
    
    def select_action(self, observation, info):
        """Select action based on simple rules"""
        my_units = info.get('my_unit_count', 0)
        enemy_units = info.get('enemy_unit_count', 0)
        cash = info.get('cash', 0)
        
        # Simple strategy:
        # 1. If we have cash and fewer units than enemy, try to produce
        # 2. If we have more units, try to attack
        # 3. Otherwise, move randomly
        
        if cash > 500 and my_units < enemy_units * 1.5:
            # Try to produce units
            action = np.array([2, 0, 0, 0, 0, 0])  # Produce infantry
        elif my_units > enemy_units and enemy_units > 0:
            # Try to attack (simplified)
            action = np.array([1, 0, 0, 0, 0, 0])  # Attack with first unit
        else:
            # Move randomly
            action = np.array([
                0,  # Move action
                random.randint(0, min(my_units-1, 99)) if my_units > 0 else 0,
                random.randint(0, 127),  # Random X
                random.randint(0, 127),  # Random Y
                0, 0
            ])
        
        self.action_history.append(action)
        return action


def agent_example():
    """Example with a simple AI agent"""
    print("🤖 Starting simple agent example...")
    
    env = create_simple_combat_env()
    agent = SimpleAgent(env)
    
    try:
        observation, info = env.reset()
        total_reward = 0
        
        for episode in range(3):  # Run 3 episodes
            print(f"\n📺 Episode {episode + 1}")
            observation, info = env.reset()
            episode_reward = 0
            
            for step in range(100):  # Max 100 steps per episode
                action = agent.select_action(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if step % 10 == 0:  # Print every 10 steps
                    print(f"  Step {step}: Reward={reward:.2f}, "
                          f"Total={episode_reward:.2f}, Units={info.get('my_unit_count', 0)}")
                
                if terminated or truncated:
                    print(f"  Episode finished at step {step}")
                    break
                
                time.sleep(0.1)  # Small delay
            
            total_reward += episode_reward
            print(f"📊 Episode {episode + 1} total reward: {episode_reward:.2f}")
        
        print(f"🏆 Average reward over 3 episodes: {total_reward / 3:.2f}")
        
    except Exception as e:
        print(f"❌ Error during agent example: {e}")
    finally:
        env.close()


# Example 3: Integration with stable-baselines3 (if available)
def sb3_training_example():
    """Example of training with stable-baselines3"""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        print("🏋️ Starting Stable-Baselines3 training example...")
        
        # Create environment
        env = create_simple_combat_env()
        
        # Check environment compatibility
        print("🔍 Checking environment compatibility...")
        check_env(env)
        print("✅ Environment is compatible with Stable-Baselines3")
        
        # Create PPO agent
        print("🧠 Creating PPO agent...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        # Train for a short time (demo purposes)
        print("🚀 Starting training...")
        model.learn(total_timesteps=5000)
        
        # Save model
        model.save("openra_ppo_demo")
        print("💾 Model saved as 'openra_ppo_demo'")
        
        # Test trained agent
        print("🎯 Testing trained agent...")
        obs, info = env.reset()
        for _ in range(20):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward:.2f}, Units: {info.get('my_unit_count', 0)}")
            
            if terminated or truncated:
                break
        
        env.close()
        
    except ImportError:
        print("❌ Stable-Baselines3 not available. Install with: pip install stable-baselines3")
    except Exception as e:
        print(f"❌ Error during SB3 example: {e}")


# Example 4: Visual environment (image observations)
def visual_example():
    """Example using image-based observations"""
    print("🖼️ Starting visual environment example...")
    
    try:
        env = create_visual_env()
        
        observation, info = env.reset()
        print(f"📊 Observation shape: {observation.shape}")
        print(f"📊 Observation type: {observation.dtype}")
        
        # Take a few random actions
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step}: Observation shape={obs.shape}, Reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error during visual example: {e}")


# Example 5: Custom reward shaping
class CustomRewardWrapper:
    """Wrapper to modify reward function"""
    
    def __init__(self, env):
        self.env = env
        self.previous_state = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_state = info.copy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Custom reward shaping
        custom_reward = reward
        
        # Bonus for maintaining units
        if self.previous_state:
            unit_preservation = info.get('my_unit_count', 0) / max(self.previous_state.get('my_unit_count', 1), 1)
            custom_reward += (unit_preservation - 1.0) * 2.0
        
        # Bonus for economic growth
        cash_growth = info.get('cash', 0) - self.previous_state.get('cash', 0) if self.previous_state else 0
        custom_reward += cash_growth / 1000.0
        
        self.previous_state = info.copy()
        
        return obs, custom_reward, terminated, truncated, info
    
    def close(self):
        self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def custom_reward_example():
    """Example with custom reward shaping"""
    print("🎛️ Starting custom reward example...")
    
    try:
        base_env = create_simple_combat_env()
        env = CustomRewardWrapper(base_env)
        
        observation, info = env.reset()
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step}: Custom reward={reward:.2f}, Cash={info.get('cash', 0)}")
            
            if terminated or truncated:
                break
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error during custom reward example: {e}")


if __name__ == "__main__":
    print("🎮 OpenRA RL Environment Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", basic_example),
        ("Simple Agent", agent_example),
        ("Stable-Baselines3 Training", sb3_training_example),
        ("Visual Environment", visual_example),
        ("Custom Reward Shaping", custom_reward_example)
    ]
    
    while True:
        print("\nAvailable examples:")
        for i, (name, _) in enumerate(examples):
            print(f"{i + 1}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect example (0-5): "))
            if choice == 0:
                break
            elif 1 <= choice <= len(examples):
                name, func = examples[choice - 1]
                print(f"\n{'=' * 20} {name} {'=' * 20}")
                func()
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print("🎯 Examples completed!")