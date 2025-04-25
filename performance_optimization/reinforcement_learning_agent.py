import random

class ReinforcementLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration rate

    def get_state_key(self, state):
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
