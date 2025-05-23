import random
import numpy as np

class ReinforcementLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01):
        self.actions = actions
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state_key(self, state):
        # Support for numpy arrays or tuples as state
        if isinstance(state, (np.ndarray, list)):
            return str(tuple(state))
        return str(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon or state_key not in self.q_table:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}

        predict = self.q_table[state_key][action]
        target = reward + self.discount_factor * max(self.q_table[next_state_key].values())
        self.q_table[state_key][action] += self.learning_rate * (target - predict)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset_parameters(self, learning_rate=None, discount_factor=None, epsilon=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if discount_factor is not None:
            self.discount_factor = discount_factor
        if epsilon is not None:
            self.epsilon = epsilon
