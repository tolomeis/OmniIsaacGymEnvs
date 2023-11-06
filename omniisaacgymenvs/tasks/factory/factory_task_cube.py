# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for cube pick task.

Inherits cube environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPick
"""

import asyncio

import hydra
import omegaconf
import os
import torch

#from omniisaacgymenvs.tasks.factory.factory_env_nut_bolt import FactoryEnvNutBolt
from omniisaacgymenvs.tasks.factory.factory_cube_env import FactoryCube
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
import omniisaacgymenvs.tasks.factory.factory_control as fc

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.transformations import *
import omni.isaac.core.utils.torch as torch_utils



class FactoryCubeTask(FactoryCube, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        super().__init__(name, sim_config, env)
        self._get_task_yaml_params()
    
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        # removed from nut
        # asset_info_path = '../tasks/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
        # self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        # self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['tasks']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskNutBoltPickPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting
        self.level = 0.0
        self.freezed_reward = torch.zeros((self.num_envs,1), device=self.device)
        self.episode_sums = {
            "dist_from_goal": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),
            "cube_vel_kickstart": torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False),

        }


    def post_reset(self):
        """
        This method is called only one time right before sim begins. 
        The 'reset' here is referring to the reset of the world, which occurs before the sim starts.
        """

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        # super().post_reset()
        self.acquire_base_tensors()
        self._acquire_task_tensors()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        # removed from nut
        # nut_grasp_heights = self.bolt_head_heights + self.nut_heights * 0.5  # nut COM
        # self.nut_grasp_pos_local = nut_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self._device).repeat(
        #     (self._num_envs, 1))
        # self.nut_grasp_quat_local = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device).unsqueeze(0).repeat(
        #     self._num_envs, 1)
        
        # Grasp pose tensors
        self.cube_grasp_pos_local = torch.tensor([0.0, 0.0, 0.01], device=self._device).repeat((self._num_envs, 1))
        self.cube_grasp_quat_local = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device).repeat((self._num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        
        self.keypoints_gripper = torch.zeros(
            (self._num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self._device
        )
        self.keypoints_cube = torch.zeros_like(self.keypoints_gripper, device=self._device)
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).unsqueeze(0).repeat(self._num_envs, 1)

    
    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            do_scale=True
        )

    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids, randomize_gripper_pose=True)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            do_scale=True,
        )


    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_object(env_ids)
        SimulationContext.step(self._env._world, render=True)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        self._reset_franka(env_ids)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
            )
            self.episode_sums[key][env_ids] = 0.0

        # self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)
        
        # step once to update physx with the newly set joint positions from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        self._reset_buffers(env_ids)
    
    async def reset_idx_async(self, env_ids, randomize_gripper_pose=True) -> None:
        """Reset specified environments."""
        print("USING ASYNC")
        self._reset_object(env_ids)
        self._reset_franka(env_ids)

        if randomize_gripper_pose:
            await self._randomize_gripper_pose_async(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
            )

        self._reset_buffers(env_ids)


    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        indices = env_ids.to(dtype=torch.int32)

        self.dof_pos[env_ids] = torch.cat(
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        self.frankas.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)
        SimulationContext.step(self._env._world, render=True)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()


        # Now compute dof pos to grasp the cube
        gripper_initial_grasp_pos = self.cube_grasp_pos[env_ids,:].clone().to(device=self.device)
        #gripper_initial_grasp_pos[:,2] -= 0.005
        gripper_initial_grasp_quat = self.cube_grasp_quat[env_ids,:].clone().to(device=self.device)

        #self.set_gripper_to(gripper_initial_grasp_pos, gripper_initial_grasp_quat)
        #dof = self._get_franka_dof(gripper_initial_grasp_pos, gripper_initial_grasp_quat,self.asset_info_franka_table.franka_gripper_width_max)

        #print(self.dof_pos[0, 0:7])
        # delta_dof = fc.compute_dof_pos_target(
        #     cfg_ctrl = self.cfg_ctrl,
        #     arm_dof_pos =                           self.dof_pos[:, 0:7],
        #     fingertip_midpoint_pos =                self.fingertip_midpoint_pos,
        #     fingertip_midpoint_quat =               self.fingertip_midpoint_quat,
        #     jacobian =                              self.fingertip_midpoint_jacobian,
        #     ctrl_target_fingertip_midpoint_pos =    gripper_initial_grasp_pos,
        #     ctrl_target_fingertip_midpoint_quat =   gripper_initial_grasp_quat,
        #     ctrl_target_gripper_dof_pos =           self.asset_info_franka_table.franka_gripper_width_max,
        #     device = self.device,
        # )

        # self.dof_pos[:, 0:7] += delta_dof

        # print('df computed by func')
        # print(dof[0, 0:7])

        # self.dof_pos[env_ids] = dof[env_ids]
        # self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        # self.frankas.set_joint_positions(positions=self.dof_pos[env_ids], indices=indices)
        # self.frankas.set_joint_position_targets(positions=self.dof_pos[env_ids], indices=indices)
        # self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        self.set_gripper_to(self.cube_grasp_pos, gripper_initial_grasp_quat, sim_steps=10)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()


    def _reset_object(self, env_ids):
        """Reset root states of cube."""
        indices = env_ids.to(dtype=torch.int32)

        # Randomize root state of cube
        cube_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        cube_noise_xy = cube_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.nut_pos_xy_initial_noise, device=self.device))
        
        self.cube_pos[env_ids, 0] = self.cfg_task.randomize.nut_pos_xy_initial[0] + cube_noise_xy[env_ids, 0]
        self.cube_pos[env_ids, 1] = self.cfg_task.randomize.nut_pos_xy_initial[1] + cube_noise_xy[env_ids, 1]
        self.cube_pos[env_ids, 2] = self.cfg_base.env.table_height + 0.005 # - self.bolt_head_heights.squeeze(-1)

        self.cube_quat[env_ids, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(len(env_ids), 1)

        self.cube_linvel[env_ids, :] = 0.0
        self.cube_angvel[env_ids, :] = 0.0

        self._cube.set_world_poses(self.cube_pos[env_ids] + self.env_pos[env_ids], self.cube_quat[env_ids], indices)
        self._cube.set_velocities(torch.cat((self.cube_linvel[env_ids], self.cube_angvel[env_ids]), dim=1), indices)
        # Goal position
        # FIXME: doesn't take into account env_ids
        self.goal_cube_pos = torch.tensor([0.0, 0.0, 0.401], device=self.device).repeat(self.num_envs,1)
        goal_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        goal_noise_xy = goal_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.goal_noise, device=self.device))
        
        self.goal_cube_pos[:, 0] += goal_noise_xy[:, 0]
        self.goal_cube_pos[:, 1] +=  goal_noise_xy[:, 1]
        sphere_pos = self.goal_cube_pos + self.env_pos
        self._sphere.set_world_poses(sphere_pos, self.cube_grasp_quat_local)
      


    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        if self.cfg_task.ctrl.control_delta_q: 
            targetq = self.dt * actions @ torch.diag(torch.tensor(self.cfg_task.rl.deltaq_action_scale, device=self.device)) + self.dof_pos
            self.frankas.set_joint_position_targets(positions=targetq)


        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        if self.cfg_task.ctrl.control_gripper:
            # Retrieve gripper DOF from actions and apply:
            gripper_actions = actions[:,12:]
            if do_scale:
                gripper_actions = gripper_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.gripper_action_scale, device=self.device))

            self.ctrl_target_gripper_dof_pos = gripper_actions
        #self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        

        self.generate_ctrl_signals()


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            # In this policy, episode length is constant
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                    self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    async def post_physics_step_async(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    await self._close_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_close_sim_steps
                    )
                    await self._lift_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps
                    )

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of cube grasping frame
        self.cube_grasp_quat, self.cube_grasp_pos = tf_combine(
            self.cube_quat,
            self.cube_pos,
            self.cube_grasp_quat_local,
            self.cube_grasp_pos_local,
        )

        # Compute pos of keypoints on gripper and cube in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = tf_combine(
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_cube[:, idx] = tf_combine(
                self.cube_grasp_quat,
                self.cube_grasp_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1)
            )[1]


    def get_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        rem_time = torch.unsqueeze(self.max_episode_length - self.progress_buf,1)
        #print(rem_time.size())
        #print(self.cube_grasp_pos.size())
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.cube_grasp_pos,
                       self.cube_grasp_quat,
                       rem_time ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        observations = {
            self.frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def calculate_metrics(self) -> None:
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()


    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
    

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        rem_time =  self.max_episode_length - self.progress_buf[0] 
        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        finger_penalty = self.ctrl_target_gripper_dof_pos * rem_time*0.05
        
        goal_pos = torch.tensor([0.0, 0.0, 0.7], device=self.device).repeat(self.num_envs,1)
        dist_penalty = torch.norm(goal_pos - self.cube_pos,dim=1)

        dist_reward = 1.0 / (1.0 + dist_penalty**2)
        #dist_reward *= dist_reward
        self.rew_buf[:] = - action_penalty * self.cfg_task.rl.action_penalty_scale 

        # Start rewarding lift in last 50 steps
        is_ending = (self.progress_buf[0] >= self.max_episode_length - 10)
        if is_ending:
            self.rew_buf[:] += dist_reward * self.cfg_task.rl.dist_reward_scale 


        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            # Check if cube is picked up and above table
            self.rew_buf[:] = torch.where(dist_penalty < 0.02, self.rew_buf[:] + 100 * torch.ones_like(self.rew_buf), self.rew_buf)
            #lift_success = self._check_lift_success(height_multiple=3.0)
            #self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(dist_penalty.float())



    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self._device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self._device) - 0.5

        return keypoint_offsets
    

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_cube - self.keypoints_gripper, p=2, dim=-1), dim=-1)
        return keypoint_dist


    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)
    

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""
        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
        
        # Step sim
        for _ in range(sim_steps):
            SimulationContext.step(self._env._world, render=True)
    

    def _lift_gripper(self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            SimulationContext.step(self._env._world, render=True)
    
    async def _close_gripper_async(self, sim_steps=20) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        await self._move_gripper_to_dof_pos_async(
            gripper_dof_pos=0.0, sim_steps=sim_steps
        )

    async def _move_gripper_to_dof_pos_async(
        self, gripper_dof_pos, sim_steps=20
    ) -> None:
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )  # No hand motion
        self._apply_actions_as_ctrl_targets(
            delta_hand_pose, gripper_dof_pos, do_scale=False
        )

        # Step sim
        for _ in range(sim_steps):
            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    async def _lift_gripper_async(
        self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20
    ) -> None:
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, franka_gripper_width, do_scale=False
            )
            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()



    def _check_lift_success(self, height_multiple):
        """Check if cube is above table by more than specified multiple times height of cube."""

        lift_success = torch.where(
            self.cube_pos[:, 2] > self.cfg_base.env.table_height + 0.7,
            # self.cube_pos[:, 2] > self.cfg_base.env.table_height + 0.05,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success
    

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # step once to update physx with the newly set joint positions from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
            device=self.device)
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            SimulationContext.step(self._env._world, render=True)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        
        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        SimulationContext.step(self._env._world, render=True)


    async def _randomize_gripper_pose_async(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""

        # step once to update physx with the newly set joint positions from reset_franka()
        await omni.kit.app.get_app().next_update_async()

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()

    def _get_franka_dof(self, current_dof_pos, target_gripper_pose, target_quat, gripper_w):
        """Set Franka DOF position target to move fingertips towards target pose."""

        target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl = self.cfg_ctrl,
            arm_dof_pos = current_dof_pos,
            fingertip_midpoint_pos = self.fingertip_midpoint_pos,
            fingertip_midpoint_quat = self.fingertip_midpoint_quat,
            jacobian = self.fingertip_midpoint_jacobian,
            ctrl_target_fingertip_midpoint_pos = target_gripper_pose,
            ctrl_target_fingertip_midpoint_quat = target_quat,
            ctrl_target_gripper_dof_pos = gripper_w,
            device = self.device,
        )
        return target_dof_pos

    def set_gripper_to(self, target_gripper_pose, target_quat, sim_steps=100):
        """Perform CLIK to move the gripper to specific pose"""
        ctrl_target_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)

        for _ in range(sim_steps):
            # Refresh data
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            # compute pose error
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos = target_gripper_pose,
                ctrl_target_fingertip_midpoint_quat = target_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Use dfferential IK to compute delta dof
            delta_arm_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method='pinv',
                jacobian=self.fingertip_midpoint_jacobian,
                device=self.device,
            )
            # Compute new DOFs
            ctrl_target_dof_pos[:, 0:7] = self.dof_pos[:,0:7] + delta_arm_dof_pos
            ctrl_target_dof_pos[:, 7:9] = self.asset_info_franka_table.franka_gripper_width_max  # gripper finger joints
            # set to robots and render
            self.frankas.set_joint_positions(positions=ctrl_target_dof_pos)
            self.frankas.set_joint_position_targets(positions=ctrl_target_dof_pos)
            self.frankas.set_joint_velocities(torch.zeros_like(self.dof_pos, device=self.device))

            SimulationContext.step(self._env._world, render=True)
            

