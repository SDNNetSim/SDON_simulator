StableBaselines3 Scripts
=========================

The `stablebaselines3` module serves as a bridge between the simulation environment and the Stable Baselines3 library,
facilitating seamless integration of deep reinforcement learning (DRL) algorithms into the simulation workflow.
Additionally, it provides interactions with the `rlzoo3` library, enhancing the versatility and capabilities of the
simulation environment.

Parameter Modification YAML Script
-----------------------------------

The module includes YAML scripts for modifying DRL algorithm parameters, enabling users to fine-tune and customize
algorithm configurations to suit specific simulation scenarios. For instance, the following YAML script `dqn.yml`
demonstrates parameter modifications for the `SimEnv` environment using the dqn algorithm:

.. code-block:: yaml

    SimEnv:
      n_timesteps: !!float 10000
      policy: 'MultiInputPolicy'
      learning_rate: !!float 0.00044
      batch_size: 32
      buffer_size: 1000000
      learning_starts: 20000
      gamma: 0.99
      target_update_interval: 5000
      train_freq: 128
      gradient_steps: -1
      exploration_fraction: 0.341
      exploration_final_eps: 0.085
      policy_kwargs: "dict(net_arch=[128, 128])"

This YAML script exemplifies how users can modify parameters such as `n_timesteps`, `learning_rate`, and
`policy_kwargs` to customize the behavior of the `SimEnv` environment for their specific needs.

Please refer to the StableBaselines3 documentation for more in-depth explanations.

.. toctree::

    register_env
