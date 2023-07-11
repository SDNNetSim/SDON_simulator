import numpy as np


class QLearning:
    # TODO: Have a way to customize discrete buckets or something like that
    def __init__(self, epsilon=0.1, episodes=1000, learn_rate=0.1, discount=0.95):
        self.done = False
        # TODO: Discretize? States and actions? Probably will end up being a numpy array
        #   If discrete, a method for this is probably a good idea
        # The Q-table with every state-action value pair
        self.q_table = np.zeros((1, 1))
        # Randomization of action selection
        self.epsilon = epsilon
        self.epsilon_decay = None
        # Amount of iterations
        self.episodes = episodes
        # Alpha in the bellman equation
        self.learn_rate = learn_rate
        # How important future actions are over current actions
        self.discount = discount

    @staticmethod
    def _environment(routed):
        # TODO: Update environment (import simulator)
        # For now, a positive one if routed, negative one if blocked
        if routed:
            reward = 1
        else:
            reward = -1

        # TODO: return an actual new state here or somewhere else?
        new_state = 1
        return reward, new_state

    def _setup_episode(self):
        # Init q-table
        self.q_table = np.zeros((1, 1))

    # TODO: Save best q-table? Or something like that
    # TODO: Training and testing (make a plan for all this, it's nonsense at the moment)
    def run(self):
        for episode in range(self.episodes):
            self._setup_episode()

            while not self.done:
                state = None

                # Choose a random action
                # TODO: Make sure this works
                if np.random.randint(0, 100) > self.epsilon:
                    # TODO: Make sure this works
                    action = self.q_table[np.random.randint(np.size(self.q_table[0]))]
                else:
                    action = np.argmax(self.q_table[state])
                reward, new_state = self._environment(routed=True)

                max_future_q = np.max(self.q_table[new_state])
                current_q = self.q_table[(state, action)]

                # Bellman equation
                new_q = ((1 - self.learn_rate) * current_q) + (
                        self.learn_rate * (reward + self.discount * max_future_q))

                # Update q-table
                self.q_table[(state, action)] = new_q


# TODO:
#  - Implement examples and run (simple implementation & custom environment)
#  - Create a plan for your q-table and environment
#  - Present plan/Begin implementing
if __name__ == '__main__':
    pass
