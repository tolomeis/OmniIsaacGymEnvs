## As far as I understand, this code wrap the main Task class with a new class that implements the feature for the demo.
# This includes the camera and video recording (?)

from omniisaacgymenvst.tasks.factory.cube_task import CubeTask

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
import numpy as np

class CubePlay(CubeTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        sim_config.task_config["env"]["numEnvs"] = 1
        CubeTask.__init__(self, name, sim_config, env)
        # set the center of the table as camera target
        self.target_table = torch.tensor([self.asset_info_franka_table.table_depth/2.0, \
                                        self.asset_info_franka_table.table_width/2.0, \
                                        self.cfg_base.env.table_height], device=self.device)
        self.create_camera()
        self.set_up_keyboard()
        self._update_camera()

        self.joint_pos_buffer = []
        self.joint_vel_buffer = []
        self.joint_ref_buffer = []
        self.gripper_vel_buffer = []
        self.lfinger_force_buffer = []
        self.rfinger_force_buffer = []
        self.joint_torque_buffer = []
        self.joint_acc_buffer = []

        self.cube_pos_buffer = []
        self.cube_quat_buffer = []
        self.cube_linvel_buffer = []
        self.cube_angvel_buffer = []
        
        self.saved_obs_vector = []
        self.saved_actions = []
        
        self.SAVE_DATA = True
        self.save_idx = 0
        self.save_ready = False

        return
        # Wrap post_reset function with camera update
    def post_reset(self):
        super().post_reset()
        self._update_camera()
        self.view_port.set_active_camera(self.camera_path)

    def get_observations(self):
        obs_vector = super().get_observations()
        if self.SAVE_DATA:
            self.joint_pos_buffer.append(self.frankas.get_joint_positions([0]).cpu().numpy())
            self.joint_vel_buffer.append(self.frankas.get_joint_velocities([0]).cpu().numpy())
            self.joint_ref_buffer.append(self.ctrl_target_dof_pos[0].cpu().numpy())
            self.lfinger_force_buffer.append(self.left_finger_force[0].cpu().numpy())
            self.rfinger_force_buffer.append(self.right_finger_force[0].cpu().numpy())
            self.gripper_vel_buffer.append(self.fingertip_midpoint_linvel.cpu().numpy())
            # self.joint_acc_buffer.append(self.frankas.get_joint_accelerations([0]).cpu().numpy())
            self.cube_pos_buffer.append(self.cube_pos.cpu().numpy())
            self.cube_quat_buffer.append(self.cube_quat.cpu().numpy())
            self.cube_linvel_buffer.append(self.cube_linvel.cpu().numpy())
            self.cube_angvel_buffer.append(self.cube_angvel.cpu().numpy())
            self.saved_obs_vector.append(obs_vector)
            self.saved_actions.append(self.actions.cpu().numpy())
            self.save_ready = True
        return obs_vector
    

    def calculate_metrics(self):

        super().calculate_metrics()


    def reset_idx(self, env_ids):
        if self.SAVE_DATA and self.save_ready:
            save_buffer = {
                "joint_pos": np.array(self.joint_pos_buffer),
                "joint_vel": np.array(self.joint_vel_buffer),
                "joint_acc": np.array(self.joint_acc_buffer),
                "joint_ref": np.array(self.joint_ref_buffer),
                "lfinger_force": np.array(self.lfinger_force_buffer),
                "rfinger_force": np.array(self.rfinger_force_buffer),
                "gripper_vel": np.array(self.gripper_vel_buffer),
                # Cube data
                "cube_pos": np.array(self.cube_pos_buffer),
                "cube_quat": np.array(self.cube_quat_buffer),
                "cube_linvel": np.array(self.cube_linvel_buffer),
                "cube_angvel": np.array(self.cube_angvel_buffer),
                "obs_vector": np.array(self.saved_obs_vector),
                "actions": np.array(self.saved_actions),

            }
            exp_name = self._cfg["experiment"]
            np.savez('exp_{0}_data_{1}'.format(exp_name, self.save_idx), **save_buffer)
            self.save_idx += 1
            self.save_ready = False
            self.joint_pos_buffer = []
            self.joint_vel_buffer = []
            self.joint_ref_buffer = []
            self.joint_torque_buffer = []
            self.gripper_vel_buffer = []
            self.lfinger_force_buffer = []
            self.rfinger_force_buffer = []
            self.joint_acc_buffer = []
            self.cube_pos_buffer = []
            self.cube_quat_buffer = []
            self.cube_linvel_buffer = []
            self.cube_angvel_buffer = []
            self.saved_obs_vector = []
            self.saved_actions = []
            print("Saving to exp_{0}_data_{1}".format(exp_name, self.save_idx))
        super().reset_idx(env_ids)


    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        self.view_port = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.view_port.set_active_camera(self.perspective_path)
    

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        # actions *= 0.0
        # actions[:, 0] -= 0.5
        super()._apply_actions_as_ctrl_targets(actions, ctrl_target_gripper_dof_pos, do_scale)


    def _update_camera(self):
        base_pos = torch.tensor([-0.6, 1.5, 0.3], device=self.device)
        base_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        camera_local_transform = torch.tensor([-1.8, 0.0, 0.6], device=self.device) 
        camera_pos = quat_apply(base_quat, camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.view_port)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(self.target_table[0].item(), self.target_table[1].item(), self.target_table[2].item())
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)
        


    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)


    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'C':
                if self.view_port.get_active_camera() == self.camera_path:
                    self.view_port.set_active_camera(self.perspective_path)
                else:
                    self.view_port.set_active_camera(self.camera_path)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pass

    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        # Every 5 steps apply a force to the cube
        # if self.progress_buf[0] % 10 == 0:
        #     is_cube_lifted = torch.where(self.cube_pos[:, 2] > self.cfg_base.env.table_height + 0.02,\
        #                               torch.ones_like(self.cube_pos[:, 2]), torch.zeros_like(self.cube_pos[:, 2]))
        #     is_cube_grasped = torch.where(torch.norm(self.cube_pos - self.fingertip_midpoint_pos, dim=1) < 0.02,\
        #                                 torch.ones_like(self.cube_pos[:, 2]), torch.zeros_like(self.cube_pos[:, 2]))
        #     random_force =  (torch.rand((self.num_envs,3), dtype=torch.float32, device=self.device) * 2.0 - 1.0) @ \
        #                     torch.diag(torch.tensor(self.cfg_task.randomize.cube_force_max, device=self.device))
        #     self._cube.apply_forces(random_force, is_cube_lifted * is_cube_grasped)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        self.previous_actions = self.actions.clone().to(self.device)
        for i in range(self.decimation):
            if self.world.is_playing():
                self._apply_actions_as_ctrl_targets(
                    actions=self.actions,
                    ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                    do_scale=True
                )
