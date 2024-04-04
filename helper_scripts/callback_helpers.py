from stable_baselines3.common.callbacks import BaseCallback


class GetModelParams(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(GetModelParams, self).__init__(verbose)

        self.model_params = None
        self.value_estimate = 0.0

    def _on_step(self) -> bool:
        self.model_params = self.model.get_parameters()

        obs = self.locals['obs_tensor']
        self.value_estimate = self.model.policy.predict_values(obs=obs)[0][0].item()
        return True
