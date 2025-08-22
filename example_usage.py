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
    """Rule-based agent: deploy MCV, then build base and infantry."""

    def __init__(self, env):
        self.env = env
        self.action_history = []
        # Build convenient mapping from action name to index
        self.action_index = {name: i for i, name in enumerate(self.env.action_types)}
        self.deployed_this_episode = False

    # --- Helpers ---
    def _find_queue_index_by_id(self, info, actor_id: int) -> int:
        ids = info.get('queue_actor_ids', [])
        try:
            return ids.index(actor_id)
        except ValueError:
            return 0

    def _find_first_index(self, pred, seq):
        for i, x in enumerate(seq):
            if pred(x):
                return i
        return -1

    def _have_building(self, info, names):
        names = set(n.lower() for n in (names if isinstance(names, (list, tuple, set)) else [names]))
        for u in info.get('my_units', []) or []:
            if u.get('type', '').lower() in names:
                return True
        return False

    def _count_buildings(self, info, names) -> int:
        names = set(n.lower() for n in (names if isinstance(names, (list, tuple, set)) else [names]))
        count = 0
        for u in info.get('my_units', []) or []:
            if u.get('type', '').lower() in names:
                count += 1
        return count

    def _queue_contains(self, queue, item_name: str) -> bool:
        for it in queue.get('Items', []) or []:
            if it.get('Item') == item_name:
                return True
        return False

    def _queue_has_done(self, queue, item_name: str) -> bool:
        for it in queue.get('Items', []) or []:
            if it.get('Item') == item_name and it.get('Done'):
                return True
        return False

    def _select_build_cell(self, info, unit_type: str):
        cells = (info.get('placeable_areas') or {}).get(unit_type, [])
        if not cells:
            return None
        # Pick a deterministic first cell for now
        return cells[0]

    def _action(self, name: str, unit_idx: int = 0, x: int = 0, y: int = 0, target_idx: int = 0, unit_type_name: str = ''):
        t = self.action_index.get(name, 0)
        unit_type_idx = self.env.unit_types.get(unit_type_name, 0)
        return np.array([t, unit_idx, x, y, target_idx, unit_type_idx], dtype=np.int64)

    # --- Core policy ---
    def select_action(self, observation, info):
        my_units_list = info.get('my_units', []) or []
        cash = info.get('cash', 0)

        # 1) Deploy MCV if present and no conyard yet
        mcv_idx = self._find_first_index(lambda u: 'mcv' in u.get('type', '').lower(), my_units_list)
        if mcv_idx >= 0 and not self._have_building(info, ['fact']) and not self.deployed_this_episode:
            if 'deploy' in self.action_index:
                action = self._action('deploy', unit_idx=mcv_idx)
                self.deployed_this_episode = True
                self.action_history.append(action)
                return action

        # Read production queues
        production = info.get('production', {}) or {}
        queues = production.get('Queues', []) or []

        # 2) If any building item finished, place it
        for q in queues:
            # Find any done building (e.g., powr, proc, tent/barr)
            for it in q.get('Items', []) or []:
                item_name = it.get('Item')
                if it.get('Done') and item_name:
                    cell = self._select_build_cell(info, item_name)
                    if cell:
                        unit_idx = self._find_queue_index_by_id(info, q.get('ActorId', 0))
                        x, y = int(cell[0]), int(cell[1])
                        action = self._action('build', unit_idx=unit_idx, x=x, y=y, unit_type_name=item_name)
                        self.action_history.append(action)
                        return action

        # 3) If any queue has buildable entries, produce in priority: powr -> tent/barr -> proc
        has_buildable = any(((q.get('Buildable', []) or [])) for q in queues)
        if has_buildable:
            barr_count_now = self._count_buildings(info, ['tent', 'barr'])
            # powr first
            act = None
            act = act or next((self._action('produce', unit_idx=self._find_queue_index_by_id(info, q.get('ActorId', 0)), unit_type_name='powr')
                               for q in queues
                               if ('powr' in [b.get('Name') for b in (q.get('Buildable', []) or [])] and not self._queue_contains(q, 'powr') and cash >= 100)), None)
            # then tent/barr under cap
            if act is None and barr_count_now < 4:
                for prefer in ('tent', 'barr'):
                    for q in queues:
                        buildable = [b.get('Name') for b in (q.get('Buildable', []) or [])]
                        if prefer in buildable and not self._queue_contains(q, prefer) and cash >= 100:
                            act = self._action('produce', unit_idx=self._find_queue_index_by_id(info, q.get('ActorId', 0)), unit_type_name=prefer)
                            break
                    if act is not None:
                        break
            # finally proc
            if act is None:
                for q in queues:
                    buildable = [b.get('Name') for b in (q.get('Buildable', []) or [])]
                    if 'proc' in buildable and not self._queue_contains(q, 'proc') and cash >= 100:
                        act = self._action('produce', unit_idx=self._find_queue_index_by_id(info, q.get('ActorId', 0)), unit_type_name='proc')
                        break
            if act is not None:
                self.action_history.append(act)
                return act

        # 4) Otherwise queue production based on economy and power:
        #    - If power is Low/Critical: build powr
        #    - If cash < 2000: build proc
        #    - Else: build tent/barr (cap at 4 total)
        power_state = (info.get('power_state') or 'Normal')
        barr_count = self._count_buildings(info, ['tent', 'barr'])

        def try_queue(target_name: str) -> np.ndarray | None:
            # Do not queue duplicates; ensure it is buildable from some queue
            for q in queues:
                if self._queue_contains(q, target_name):
                    continue
                buildable = [b.get('Name') for b in (q.get('Buildable', []) or [])]
                if target_name in buildable and cash >= 100:
                    unit_idx = self._find_queue_index_by_id(info, q.get('ActorId', 0))
                    return self._action('produce', unit_idx=unit_idx, unit_type_name=target_name)
            return None

        # Power first
        if power_state in ('Low', 'Critical') and not self._queue_contains({'Items': []}, 'powr'):
            act = try_queue('powr')
            if act is not None:
                self.action_history.append(act)
                return act

        # Economy-driven building
        if cash < 2000:
            act = try_queue('proc')
            if act is not None:
                self.action_history.append(act)
                return act
        else:
            # Build barracks if under cap
            if barr_count < 4:
                # Prefer tent then barr depending on faction availability
                act = try_queue('tent') or try_queue('barr')
                if act is not None:
                    self.action_history.append(act)
                    return act

        # 5) After barracks count >= 1, constantly produce infantry (e1)
        if self._have_building(info, ['tent', 'barr']):
            # Always attempt to queue e1 if any barracks queue can build it
            for q in queues:
                buildable = [b.get('Name') for b in (q.get('Buildable', []) or [])]
                if 'e1' in buildable and cash >= 100:
                    unit_idx = self._find_queue_index_by_id(info, q.get('ActorId', 0))
                    action = self._action('produce', unit_idx=unit_idx, unit_type_name='e1')
                    self.action_history.append(action)
                    return action

        # 6) If we have ongoing production but nothing to place yet, wait
        for q in queues:
            items = q.get('Items', []) or []
            if any((not it.get('Done', False)) for it in items):
                action = self._action('noop')
                self.action_history.append(action)
                return action

        # Fallback: idle (no-op move to current position of first unit if exists)
        if my_units_list:
            u0 = my_units_list[0]
            action = self._action('noop')
        else:
            action = self._action('noop')

        self.action_history.append(action)
        return action


def agent_example():
    """Example with a simple AI agent"""
    print("🤖 Starting simple agent example...")
    
    # Enable full action set including noop/produce/build/deploy
    env = OpenRAEnvironment(
        api_port=8082,
        observation_type="vector",
        enable_actions=['noop', 'move', 'attack', 'produce', 'build', 'deploy']
    )
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
                # Print selected action each step
                try:
                    atype_idx = int(action[0])
                    atype = env.action_types[atype_idx] if 0 <= atype_idx < len(env.action_types) else str(atype_idx)
                    unit_idx = int(action[1])
                    tx, ty = int(action[2]), int(action[3])
                    target_idx = int(action[4])
                    utype_idx = int(action[5])
                    utype_name = env.reverse_unit_types.get(utype_idx, str(utype_idx))
                    print(f"  Step {step} action: type={atype}, unit_idx={unit_idx}, target=({tx},{ty}), target_idx={target_idx}, unit_type={utype_name}")
                except Exception:
                    print(f"  Step {step} action (raw): {action}")
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


def build_with_placeable_example():
    """Example: read Production + PlaceableAreas, then build a structure at a valid cell."""
    print("🏗️ Starting build-with-placeable example...")
    env = OpenRAEnvironment(api_port=8082, observation_type="vector", enable_actions=['move','attack','produce','build','deploy'])
    try:
        obs, info = env.reset()
        prod = info.get('production', {})
        # Pick a queue that has finished items
        target_q = None
        target_build = None
        for q in prod.get('Queues', []):
            # Look for finished items in Items
            for it in q.get('Items', []):
                if it.get('Done') and it.get('Item'):
                    target_q = q
                    target_build = it['Item']
                    break
            if target_q:
                break
        if not target_q or not target_build:
            print("No finished build items found; try queueing production first.")
            return
        # Lookup placeable area for this unit type
        placeable = info.get('placeable_areas', {})
        coords = placeable.get(target_build, [])
        if not coords:
            print("No valid placement cells for", target_build)
            return
        tx, ty = coords[0]
        # Build action: [action_type(build), unit_idx(any my unit index mapped to producer), tx, ty, target_id, unit_type_idx]
        # We will pass unit_idx=0 and rely on server ActorId mapping by selecting the right ActorId in env (advanced users can add producer mapping)
        unit_type_idx = next((idx for idx, name in env.reverse_unit_types.items() if name == target_build), 0)
        action = np.array([env.action_types.index('build'), 0, tx, ty, 0, unit_type_idx], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Issued build for", target_build, "at", (tx, ty), "reward=", reward)
    finally:
        env.close()

if __name__ == "__main__":
    print("🎮 OpenRA RL Environment Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", basic_example),
        ("Simple Agent", agent_example),
        ("Stable-Baselines3 Training", sb3_training_example),
        ("Visual Environment", visual_example),
        ("Custom Reward Shaping", custom_reward_example),
        ("Build using PlaceableAreas", build_with_placeable_example)
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