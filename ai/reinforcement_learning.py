class QLearning:
    # TODO: Have a way to customize discrete buckets or something like that
    def __init__(self, epsilon=0.01):
        self.done = False
        # TODO: Discretize? States and actions?
        # The Q-table with every state-action value pair
        self.table = None
        # Randomization of action selection
        self.epsilon = epsilon

    def _environment(self):
        pass

    def run(self):
        pass

    # TODO: Transfer this to the init, it is your setup
    def setup(self):
        # TODO: Update to a custom environment
        env = None
        # TODO: Define actions
        actions = None

        while not self.done:
            # TODO: Update, this will be my environment function I believe
            new_state, reward, self.done = None


# TODO:
#  - Implement examples and run (simple implementation & custom environment)
#  - Create a plan for your q-table and environment
#  - Present plan/Begin implementing
if __name__ == '__main__':
    pass
