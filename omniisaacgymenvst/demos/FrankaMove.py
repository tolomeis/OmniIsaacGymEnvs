
from omniisaacgymenvst.tasks.factory_task_cube import FactoryCubeTask


from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import tf_combine

import numpy as np
import torch
import math

import omni
import carb

from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.kit.viewport.utility import get_viewport_from_window_name
from pxr import Sdf


class FrankaMoveDemo(FactoryCubeTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        
        sim_config.task_config["env"]["learn"]["episodeLength_s"] = 120
        FactoryCubeTask.__init__(self, name, sim_config, env)

        # self._prim_selection = omni.usd.get_context().get_selection()
        # self._selected_id = None
        # self._previous_selected_id = None
        return

    def pre_physics_step(self, actions) -> None:
        # Increase the target distance
        self.target_distance += 0.01  # Adjust this value as needed

        # Calculate the new target position
        target_position = self.robot_position + [self.target_distance, 0, 0]  # Adjust this as needed

        # Set the new target position
        self.env.set_target_position(target_position)

        # Call the parent class's pre_physics_step method
        super().pre_physics_step(actions)

    
    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):

        dir = (self.goal_cube_pos + self.cube_grasp_pos ) / 2.0
        self.ctrl_target_fingertip_midpoint_pos = dir
        self.ctrl_target_fingertip_midpoint_quat = self.cube_grasp_quat.clone().to(device=self.device)

     
        self._gripper_cyl.set_world_poses(self.ctrl_target_fingertip_midpoint_pos + self.env_pos,
                                     self.ctrl_target_fingertip_midpoint_quat,
                                    torch.arange(self._num_envs, dtype=torch.int64, device=self._device))
            
                        

        self.generate_ctrl_signals()