from gymnasium.envs.mujoco.ant_v4 import AntEnv
import numpy as np
from gymnasium import utils
import os
from scipy import ndimage
from scipy.signal import convolve2d
from gymnasium import spaces
import gymnasium as gym
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 10.0,
}
class QuantrupedEnv(AntEnv):
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=0.0,
        render_mode = None,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        super().__init__(
            xml_file=os.path.join(os.path.dirname(__file__), 'assets', 'ant.xml'),
            ctrl_cost_weight=ctrl_cost_weight,
            use_contact_forces=False,
            contact_cost_weight=contact_cost_weight,
            render_mode = render_mode,
            **kwargs
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
        )        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float64
        )

        ant_mass = mujoco.mj_getTotalmass(self.model)
        mujoco.mj_setTotalmass(self.model, ant_mass *  10)

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)