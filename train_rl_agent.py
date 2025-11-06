#!/usr/bin/env python3
"""
OWLBAN GROUP - Reinforcement Learning Agent Training Script
Trains NVIDIA GPU-accelerated RL agents for optimization tasks
"""

import sys
import os
import logging
import numpy as np
import time
from performance_optimization.reinforcement_learning_agent import ReinforcementLearningAgent

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rl_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_sample_environment():
    """Create a sample optimization environment"""
    class SimpleOptimizationEnv:
        def __init__(self):
            self.state_size = 4
            self.action_space = ['increase', 'decrease', 'maintain', 'optimize']
            self.max_steps = 100
            self.current_step = 0
            self.optimal_value = 10.0
            self.current_value = 5.0

        def reset(self):
            self.current_step = 0
            self.current_value = np.random.uniform(0, 20)
            return [self.current_value, self.optimal_value - self.current_value,
                   self.current_step / self.max_steps, np.random.random()]

        def step(self, action):
            self.current_step += 1

            # Simple reward function
            if action == 'optimize':
                reward = 1.0 if abs(self.current_value - self.optimal_value) < 2 else -0.1
                self.current_value += np.random.normal(0, 0.5)
            elif action == 'increase':
                self.current_value += 0.5
                reward = 0.1 if self.current_value < self.optimal_value else -0.2
            elif action == 'decrease':
                self.current_value -= 0.5
                reward = 0.1 if self.current_value > self.optimal_value else -0.2
            else:  # maintain
                reward = -0.05

            # Keep value in bounds
            self.current_value = np.clip(self.current_value, 0, 20)

            done = self.current_step >= self.max_steps
            next_state = [self.current_value, self.optimal_value - self.current_value,
                         self.current_step / self.max_steps, np.random.random()]

            return next_state, reward, done

    return SimpleOptimizationEnv()

def train_rl_agent():
    """Train the reinforcement learning agent"""
    logger = logging.getLogger("RLTraining")

    # Create environment and agent
    env = create_sample_environment()
    agent = ReinforcementLearningAgent(
        actions=env.action_space,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=0.3,
        use_gpu=True
    )

    logger.info("Starting RL agent training...")
    logger.info(f"Actions: {env.action_space}")
    logger.info(f"GPU available: {agent.device}")

    # Training parameters
    num_episodes = 100
    max_steps_per_episode = 50

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Agent chooses action
            action = agent.choose_action(state)

            # Environment responds
            next_state, reward, done = env.step(action)

            # Agent learns
            agent.learn(state, action, reward, next_state)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.3f}")

    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Final average reward: {np.mean(episode_rewards):.3f}")
    logger.info(f"Best episode reward: {max(episode_rewards):.3f}")

    # Get GPU status
    gpu_status = agent.get_gpu_status()
    logger.info(f"GPU Status: {gpu_status}")

    return {
        "episodes": num_episodes,
        "avg_reward": np.mean(episode_rewards),
        "best_reward": max(episode_rewards),
        "gpu_status": gpu_status,
        "training_time": time.time()
    }

def main():
    """Main training function"""
    setup_logging()
    logger = logging.getLogger("RLTraining")

    try:
        logger.info("OWLBAN GROUP - RL Agent Training Started")
        results = train_rl_agent()

        logger.info("Training Results:")
        logger.info(f"- Episodes: {results['episodes']}")
        logger.info(f"- Average Reward: {results['avg_reward']:.3f}")
        logger.info(f"- Best Reward: {results['best_reward']:.3f}")
        logger.info(f"- GPU Memory: {results['gpu_status']['memory_allocated']:.2f} GB")

        # Save results
        with open('rl_training_results.json', 'w') as f:
            import json
            json.dump(results, f, indent=2)

        logger.info("RL training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
