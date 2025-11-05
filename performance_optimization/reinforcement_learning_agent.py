import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Optional, Tuple

# NVIDIA-specific imports
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cp = None
    cupy_available = False

try:
    import tensorrt as trt
    tensorrt_available = True
except ImportError:
    trt = None
    tensorrt_available = False

class OptimizedDQNNetwork(nn.Module):
    """NVIDIA-optimized Deep Q-Network with cuDNN acceleration"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(OptimizedDQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # cuDNN optimized
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.layers(x)

class ReinforcementLearningAgent:
    def __init__(self, actions, learning_rate=0.001, discount_factor=0.99, epsilon=0.2,
                 epsilon_decay=0.995, epsilon_min=0.01, use_gpu=True):
        self.actions = actions
        self.action_size = len(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.logger = logging.getLogger("RLAgent")
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    self.logger.info("NVIDIA GPU-accelerated RL using device: %s", self.device)

        # Initialize traditional Q-table as fallback
        self.q_table = {}

        # Initialize Deep Q-Network for GPU acceleration
        self.state_size = 10  # Default, will be updated dynamically
        self.dqn = OptimizedDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn = OptimizedDQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Enable cuDNN optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64

        self.update_target_every = 100
        self.step_count = 0

    def get_state_key(self, state):
        """Convert state to string key for Q-table"""
        if isinstance(state, (np.ndarray, list, tuple)):
            return str(tuple(state))
        return str(state)

    def update_state_size(self, state):
        """Dynamically update state size based on input"""
        if isinstance(state, (list, tuple, np.ndarray)):
            new_size = len(state)
        else:
            new_size = 1

        if new_size != self.state_size:
            self.state_size = new_size
            self.logger.info("Updating DQN state size to %d", self.state_size)
            # Reinitialize networks with new state size
            self.dqn = OptimizedDQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_dqn = OptimizedDQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_dqn.load_state_dict(self.dqn.state_dict())
            self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

    def choose_action(self, state):
        """Choose action using NVIDIA GPU-accelerated DQN"""
        self.update_state_size(state)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get Q-values from DQN
        with torch.no_grad():
            q_values = self.dqn(state_tensor)
            action_idx = torch.argmax(q_values).item()

        return self.actions[action_idx]

    def learn(self, state, action, reward, next_state):
        """Learn using NVIDIA GPU-accelerated experience replay"""
        self.update_state_size(state)

        # Store experience in replay buffer
        self.memory.append((state, self.actions.index(action), reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        # Update traditional Q-table as fallback
        self._update_q_table(state, action, reward, next_state)

        # Train DQN if enough experiences
        if len(self.memory) >= self.batch_size:
            self._train_dqn()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _update_q_table(self, state, action, reward, next_state):
        """Update traditional Q-table"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}

        predict = self.q_table[state_key][action]
        target = reward + self.discount_factor * max(self.q_table[next_state_key].values())
        self.q_table[state_key][action] += self.learning_rate * (target - predict)

    def _train_dqn(self):
        """Train DQN using NVIDIA GPU acceleration"""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # Current Q values
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + self.discount_factor * next_q

        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_parameters(self, learning_rate=None, discount_factor=None, epsilon=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if discount_factor is not None:
            self.discount_factor = discount_factor
        if epsilon is not None:
            self.epsilon = epsilon

    def get_gpu_status(self):
        """Get NVIDIA GPU status for RL training"""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0,
            "memory_reserved": torch.cuda.memory_reserved(self.device) / 1024**3 if torch.cuda.is_available() else 0,
            "experience_buffer_size": len(self.memory)
        }
