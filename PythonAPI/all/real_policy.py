import os.path
import time

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet_policy import \
    PointNavResNetPolicy
from habitat_baselines.rl.ddppo.policy.splitnet_policy import \
    PointNavSplitNetPolicy
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy, PointNavContextPolicy, PointNavContextCMAPolicy
from habitat_baselines.utils.common import batch_obs
from habitat.config import Config


# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class RealPolicy:
    def __init__(self, cfg, observation_space, action_space, device):
        self.device = device
        checkpoint = torch.load(cfg.weights, map_location="cpu")
        config = checkpoint["config"]
        if "num_cnns" not in config.RL.POLICY:
            config.RL.POLICY["num_cnns"] = 1
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.freeze()
        self.policy = eval(cfg.policy_name).from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)
        # Load trained weights into the policy

        # If using Splitnet policy, filter out decoder stuff, as it's not used at test-time
        self.policy.load_state_dict(
            {k[len("actor_critic."):]: v for k, v in checkpoint["state_dict"].items()},
            # strict=True,
            strict=False,
        )

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, observations, deterministic=True):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            start_time = time.time()
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=deterministic,
            )
            inf_time = time.time() - start_time
            print(f"Inference time: {inf_time}")
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions


class NavPolicy(RealPolicy):
    def __init__(self, cfg, device):
        obs_right_key = f"spot_right_{cfg.sensor_type}"
        obs_left_key = f"spot_left_{cfg.sensor_type}"
        observation_space = SpaceDict(
            {
                obs_left_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                obs_right_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_dim = 3 if cfg.use_horizontal_velocity else 2
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (action_dim,))
        action_space.n = action_dim
        super().__init__(
            cfg,
            observation_space,
            action_space,
            device,
        )


class ContextNavPolicy(RealPolicy):
    def __init__(self, cfg, device):
        if cfg.sensor_type == "depth":
            obs_right_key = "spot_right_depth"
            obs_left_key = "spot_left_depth"
        elif cfg.sensor_type == "gray":
            obs_right_key = "spot_right_gray"
            obs_left_key = "spot_left_gray"
        context_key = f"context_{cfg.context_type}"
        map_dim = 2 if cfg.use_agent_map else 1
        context_shape = (2,) if context_key == "context_waypoint" else (cfg.map_resolution, cfg.map_resolution, map_dim)
        observation_space = SpaceDict(
            {
                obs_left_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                obs_right_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                context_key: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=context_shape,
                    dtype=np.float32,
                ),
            }
        )
        action_dim = 3 if cfg.use_horizontal_velocity else 2
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (action_dim,))
        action_space.n = action_dim
        super().__init__(
            cfg,
            observation_space,
            action_space,
            device,
        )


if __name__ == "__main__":
    nav_policy = NavPolicy(
        "weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth",
        device="cpu",
    )
    nav_policy.reset()
    observations = {
        "spot_left_depth": np.zeros([256, 128, 1], dtype=np.float32),
        "spot_right_depth": np.zeros([256, 128, 1], dtype=np.float32),
        "pointgoal_with_gps_compass": np.zeros(2, dtype=np.float32),
    }
    actions = nav_policy.act(observations)
    print("actions:", actions)