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
    "distance": 100.0,
    "type": 1,
    "trackbodyid": 0,
    "elevation": 300.0,
}

def create_new_hfield(mj_model, smoothness = 0.15, bump_scale=2.):
    # Generation of the shape of the height field is taken from the dm_control suite,
    # see dm_control/suite/quadruped.py in the escape task (but we don't use the bowl shape).
    # Their parameters are TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
    # and TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters). 
    res = mj_model.hfield_ncol[0]
    row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
    # Random smooth bumps.
    terrain_size = 2 * mj_model.hfield_size[0, 0]
    bump_res = int(terrain_size / bump_scale)
    bumps = np.random.uniform(smoothness, 1, (bump_res, bump_res))
    smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
    # Terrain is elementwise product.
    hfield = (smooth_bumps - np.min(smooth_bumps))[0:mj_model.hfield_nrow[0],0:mj_model.hfield_ncol[0]]
    # Clears a patch shaped like box, assuming robot is placed in center of hfield.
    # Function was implemented in an old rllab version.
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    patch_size = 2
    fromrow, torow = h_center - int(0.5*patch_size), h_center + int(0.5*patch_size)
    fromcol, tocol = w_center - int(0.5*patch_size), w_center + int(0.5*patch_size)
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((patch_size,patch_size)) / patch_size**2
    s = convolve2d(hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)], K, mode='same', boundary='symm')
    hfield[fromrow-(patch_size-1):torow+(patch_size-1), fromcol-(patch_size-1):tocol+(patch_size-1)] = s
    # Last, we lower the hfield so that the centre aligns at zero height
    # (importantly, we use a constant offset of -0.5 for rendering purposes)
    #print(np.min(hfield), np.max(hfield))
    hfield = hfield - np.max(hfield[fromrow:torow, fromcol:tocol])
    mj_model.hfield_data[:] = hfield.ravel()

class QuantrupedEnv(AntEnv):
    def __init__(
        self,
        xml_file="ant_hfield.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=0.0,
        render_mode = None,
        terminate_when_unhealthy=True,
        healthy_z_range=(-5, 5),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        hf_smoothness=0.25,
        **kwargs
    ):
        super().__init__(
            xml_file=os.path.join(os.path.dirname(__file__), 'assets', 'ant_hfield.xml'),
            ctrl_cost_weight=ctrl_cost_weight,
            use_contact_forces=False,
            contact_cost_weight=contact_cost_weight,
            render_mode = render_mode,
            **kwargs
        )

        self._healthy_z_range = healthy_z_range

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float64
        )        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float64
        )

        ant_mass = mujoco.mj_getTotalmass(self.model)
        mujoco.mj_setTotalmass(self.model, ant_mass *  10)

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    def update_hfield(self, smoothness=1, bump_scale=2):
        if smoothness is not None:
            self.smoothness = smoothness
        if bump_scale is not None:
            self.bump_scale = bump_scale
        create_new_hfield(self.model, self.smoothness, self.bump_scale)
        #print("Hfield updated with smoothness: ", self.smoothness, " and bump scale: ", self.bump_scale)