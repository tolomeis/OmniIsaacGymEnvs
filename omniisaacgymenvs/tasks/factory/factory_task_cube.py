# Copyright (c) 2023, Simone Tolomei

""" Time-Driven Manipulation

This script implements a time-driven manipulation task for the Franka Panda robot in Isaac Gym. 
The robot learns to perform a movement task with a cube. The cube is spawned at a random
location on the table and the robot has to pick it up and place it at a target location.
The training script doesn't enforce any pick-and-place behavior, but the robot learns it
by itself. The task is considered successful if the cube is within a specific distance from
the target location at the end of the episode.

To run this script, execute the following command from the root of the repository:
 /isaac-sim/python.sh omniisaacgymenvs/scripts/rlgames_train.py task=FactoryCube
"""


import hydra
import omegaconf
import os
import torch

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
        self.max_episode_length = self.cfg_task.rl.max_episode_length  

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
        # Grasp pose tensors
        self.cube_grasp_pos_local = torch.tensor([0.0, 0.0, 0.01], device=self._device).repeat((self._num_envs, 1))
        self.cube_grasp_quat_local = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device).repeat((self._num_envs, 1))
        self.goal_cube_pos = torch.tensor([0.0, 0.0, 0.401], device=self.device).repeat(self.num_envs,1)
        self.cube_pos_initial = torch.tensor([0.0, 0.0, 0.0,], device=self.device).repeat(self.num_envs,1)

    
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

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_object(env_ids)
        self._reset_task()
        SimulationContext.step(self._env._world, render=True)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self._reset_franka(env_ids)
        # self._randomize_gripper_pose(env_ids, sim_steps=10)

        # Reset buffers and metrics
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
            )
            self.episode_sums[key][env_ids] = 0.0

        
        # step once to update physx with the newly set joint positions from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        self._reset_buffers(env_ids)
    
                      
    def _reset_task(self):
        """
        Resets the task by initializing the maximum episode length and randomizing it.
        """
        self.max_episode_length = self.cfg_task.rl.max_episode_length
        episode_lenght_noise = torch.randint(
            - self.cfg_task.randomize.ep_lenght_noise,
              self.cfg_task.randomize.ep_lenght_noise,
            size=(1,1),dtype=torch.int32, device=self.device).item()
        # episode_lenght_noise = torch_utils.np.random.randint(-self.cfg_task.randomize.ep_lenght_noise, 
        #                                          self.cfg_task.randomize.ep_lenght_noise)
        self.max_episode_length += episode_lenght_noise



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
        gripper_initial_grasp_quat = self.cube_grasp_quat[env_ids,:].clone().to(device=self.device)
        # gripper_initial_grasp_quat[:,2] += 0.01

        # move gripper to grasp pose with CLIK
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
            torch.tensor(self.cfg_task.randomize.cube_pos_xy_initial_noise, device=self.device))
        
        self.cube_pos[env_ids, 0]   = self.cfg_task.randomize.cube_pos_xy_initial[0] + cube_noise_xy[env_ids, 0]
        self.cube_pos[env_ids, 1]   = self.cfg_task.randomize.cube_pos_xy_initial[1] + cube_noise_xy[env_ids, 1]
        self.cube_pos[env_ids, 2]   = self.cfg_base.env.table_height + 0.005 
        self.cube_pos_initial = self.cube_pos.clone().to(device=self.device)
        self.cube_quat[env_ids, :]  = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(len(env_ids), 1)

        self.cube_linvel[env_ids, :] = 0.0
        self.cube_angvel[env_ids, :] = 0.0

        self._cube.set_world_poses(self.cube_pos[env_ids] + self.env_pos[env_ids], self.cube_quat[env_ids], indices)
        self._cube.set_velocities(torch.cat((self.cube_linvel[env_ids], self.cube_angvel[env_ids]), dim=1), indices)


        # Randomize goal position
        goal_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        goal_noise_xy = goal_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.goal_noise_xy, device=self.device))
        
        self.goal_cube_pos = torch.tensor(self.cfg_task.randomize.goal_initial_pose, device=self.device).repeat(self.num_envs,1)
        self.goal_cube_pos[env_ids, 0] += goal_noise_xy[env_ids, 0]
        self.goal_cube_pos[env_ids, 1] +=  goal_noise_xy[env_ids, 1]
        self._sphere.set_world_poses(self.goal_cube_pos[env_ids] + self.env_pos[env_ids], self.cube_grasp_quat_local[env_ids], indices)
      


    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """
        Applies actions from policy as position/rotation targets.

        Args:
            actions (torch.Tensor): The actions to apply as position/rotation targets.
            ctrl_target_gripper_dof_pos (torch.Tensor): The gripper DOF positions to apply.
            do_scale (bool): Whether to scale the actions.

        Returns:
            None
        """

        # if self.cfg_task.ctrl.control_delta_q: 
        #     targetq = self.dt * actions @ torch.diag(torch.tensor(self.cfg_task.rl.deltaq_action_scale, device=self.device)) + self.dof_pos
        #     self.frankas.set_joint_position_targets(positions=targetq)


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

        # if self.cfg_ctrl['do_force_ctrl']:
        #     # Interpret actions as target forces and target torques
        #     force_actions = actions[:, 6:9]
        #     if do_scale:
        #         force_actions = force_actions @ torch.diag(
        #             torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

        #     torque_actions = actions[:, 9:12]
        #     if do_scale:
        #         torque_actions = torque_actions @ torch.diag(
        #             torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

        #     self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        if self.cfg_task.ctrl.control_gripper:
            # Retrieve gripper DOF from actions and apply:
            gripper_actions = actions[:,6:]
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
        





    def get_observations(self):
        """Compute observations."""
        # Shallow copies of tensors
        rem_time = torch.unsqueeze(self.max_episode_length - self.progress_buf,1)
        d_to_goal = (self.goal_cube_pos - self.cube_pos)
        d_to_cube = (self.fingertip_midpoint_pos - self.cube_grasp_pos)
        # normalized_fingertip_midpoint_pos = 
        # normalized_fingertip_midpoint_quat =
        # normalized_fingertip_midpoint_linvel =
        # normalized_fingertip_midpoint_angvel = 
        # normalized_cube_grasp_pos =
        # normalized_cube_grasp_quat =
        
        obs_tensors = [self.fingertip_midpoint_pos,
                        self.fingertip_midpoint_quat,
                        self.fingertip_midpoint_linvel,
                        self.fingertip_midpoint_angvel,
                        self.cube_grasp_pos,
                        self.cube_grasp_quat,
                        rem_time,
                        d_to_goal,
                        self.max_episode_length]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  

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
        #rem_time =  self.max_episode_length - self.progress_buf[0] 
        
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        self.rew_buf[:] = - action_penalty * self.cfg_task.rl.action_penalty_scale 
        
        # Normalize dist_penalty with respect to initial distance
        initial_dist = torch.norm(self.goal_cube_pos - self.cube_pos_inital, dim=1)
        dist_penalty = torch.norm(self.goal_cube_pos - self.cube_pos,dim=1) / initial_dist
        dist_reward = (1.0 / (1.0 + dist_penalty**2))**2
        
        # Start rewarding lift in last 10 steps
        is_ending = (self.progress_buf[0] >= self.max_episode_length - 10)
        if is_ending:
            self.rew_buf[:] += dist_reward * self.cfg_task.rl.goal_scale 
            self.episode_sums['dist_from_goal'] += dist_reward * self.cfg_task.rl.goal_scale 

        if (self.level <= 5.0):
            self.freezed_reward = torch.norm(self.cube_linvel, dim=1)*2.0


        self.rew_buf[:] += self.freezed_reward
        self.episode_sums['cube_vel_kickstart'] += self.freezed_reward
        
        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)


        if is_last_step:
            # Check if cube is picked up and above table
            self.rew_buf[:] = torch.where(dist_penalty < 0.02, self.rew_buf[:] + 100 * torch.ones_like(self.rew_buf), self.rew_buf)
            lift_success = self._check_lift_success(height_multiple=1.0)
            #self.level += torch.mean(lift_success.float())
            self.level += torch.mean(torch.norm(self.cube_linvel, dim=1))
            self.extras['final_mean_dists'] = torch.mean(dist_penalty.float())
            self.extras['cube_lifted'] = torch.mean(lift_success.float())
            self.extras['level'] = self.level
            


    def _check_lift_success(self, height_multiple):
        """Check if cube is above table by more than specified multiple times height of cube."""

        lift_success = torch.where(
            self.cube_pos[:, 2] > self.cfg_base.env.table_height + height_multiple * 0.02,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success
    

    def _randomize_gripper_pose(self, sim_steps=5):
        """
        Adds a random offset to the current gripper pose.

        Args:
            sim_steps (int): Number of simulation steps to perform.

        Returns:
            None
        """
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos.clone().to(device=self.device)

        fingertip_midpoint_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
            device=self.device)
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = self.fingertip_midpoint_quat.clone().to(device=self.device)

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

        self.dof_vel[:, :] = torch.zeros_like(self.dof_vel[:])
        idx = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.frankas.set_joint_velocities(self.dof_vel[:], indices=idx)

        # step once to update physx with the newly set joint velocities
        SimulationContext.step(self._env._world, render=True)



    def set_gripper_to(self, target_gripper_pose, target_quat, sim_steps=100):
        """
        Moves the gripper to a specific pose using CLIK (Closed-Loop Inverse Kinematics).

        Args:
            target_gripper_pose (torch.Tensor): The desired position of the gripper in the world frame, as a 3D tensor.
            target_quat (torch.Tensor): The desired orientation of the gripper in the world frame, as a quaternion tensor.
            sim_steps (int, optional): The number of simulation steps to perform. Defaults to 100.

        Returns:
            None
        """
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
            
