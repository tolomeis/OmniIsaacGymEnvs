


import carb
import hydra
import math
import numpy as np
import torch

from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.objects import FixedCylinder
from omni.isaac.core.objects import GroundPlane

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvst.tasks.base.rl_task import RLTask
from omniisaacgymenvst.robots.articulations.franka_softclaw import FrankaSoftclaw
from omni.isaac.core.materials import PhysicsMaterial


from pxr import PhysxSchema, UsdPhysics
import omniisaacgymenvst.tasks.factory.softclaw_franka_control as fc
from omniisaacgymenvst.tasks.factory.factory_schema_class_base import FactoryABCBase

from omniisaacgymenvst.tasks.factory.factory_schema_config_base import (
    FactorySchemaConfigBase,
)

class SoftclawCubeBase(RLTask, FactoryABCBase):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize instance variables. Initialize RLTask superclass."""

        # Set instance variables from base YAML
        self._get_base_yaml_params()
        self._env_spacing = self.cfg_base.env.env_spacing

        # Set instance variables from task and train YAMLs
        self._sim_config = sim_config
        self._cfg = sim_config.config  # CL args, task config, and train config
        self._task_cfg = sim_config.task_config  # just task config
        self._num_envs = sim_config.task_config["env"]["numEnvs"]
        self._num_observations = sim_config.task_config["env"]["numObservations"]
        self._num_actions = sim_config.task_config["env"]["numActions"]

        super().__init__(name, env)

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        config_path = (
            "task/SoftclawCubeBase.yaml"  # relative to Gym's Hydra search path (cfg dir)
        )
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/softclaw_cube_asset_info.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table[""][""][""][
            "tasks"
        ]["factory"][
            "yaml"
        ]  # strip superfluous nesting

    def import_franka_assets(self, add_to_stage=True):
        """Set Franka and table asset options. Import assets."""

        self._stage = get_current_stage()

        if add_to_stage:
            franka_translation = np.array([self.cfg_base.env.franka_depth, 0.0, 0.0])
            franka_orientation = np.array([1.0, 0.0, 0.0, 0.0])

            franka = FrankaSoftclaw(
                prim_path=self.default_zero_env_path + "/franka",
                name="franka",
                translation=franka_translation,
                orientation=franka_orientation,
            )
            self._sim_config.apply_articulation_settings(
                "franka",
                get_prim_at_path(franka.prim_path),
                self._sim_config.parse_actor_config("franka"),
            )

            for link_prim in franka.prim.GetChildren():
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(
                        self._stage, link_prim.GetPrimPath()
                    )
                    rb.GetDisableGravityAttr().Set(False)
                    rb.GetRetainAccelerationsAttr().Set(False)
                    if self.cfg_base.sim.add_damping:
                        rb.GetLinearDampingAttr().Set(
                            1.0
                        )  # default = 0.0; increased to improve stability
                        rb.GetMaxLinearVelocityAttr().Set(
                            1.0
                        )  # default = 1000.0; reduced to prevent CUDA errors
                        rb.GetAngularDampingAttr().Set(
                            5.0
                        )  # default = 0.5; increased to improve stability
                        rb.GetMaxAngularVelocityAttr().Set(
                            2 / math.pi * 180
                        )  # default = 64.0; reduced to prevent CUDA errors
                    else:
                        rb.GetLinearDampingAttr().Set(0.0)
                        rb.GetMaxLinearVelocityAttr().Set(1000.0)
                        rb.GetAngularDampingAttr().Set(0.5)
                        rb.GetMaxAngularVelocityAttr().Set(64 / math.pi * 180)


            plane = GroundPlane(prim_path="/World/GroundPlane", z_position=0)
        self.parse_controller_spec(add_to_stage=add_to_stage)

    def acquire_base_tensors(self):
        """Acquire tensors."""

        self.num_dofs = 8
        self.env_pos = self._env_pos

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_torque = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.ctrl_target_dof_pos = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.ctrl_target_gripper_dof_pos = torch.zeros(
            (self.num_envs, 1), device=self.device
        )
        self.ctrl_target_fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

        self.collision_count = torch.zeros(
            (self.num_envs, 1), device=self.device
        )

        self.franka_gravity_torque = torch.zeros(
            (self.num_envs, 7), device=self.device
        )

        self.arm_dof_acc = torch.zeros(
            (self.num_envs, 7), device=self.device
        )

        self.franka_acc_limits = torch.tensor([15,7.5,10,12.5,15,20,20], device=self.device)



    def refresh_base_tensors(self):
        """Refresh tensors."""

        if not self.world.is_playing():
            return

        self.dof_pos = self.frankas.get_joint_positions(clone=False)

        vel = self.frankas.get_joint_velocities(clone=False)
        self.arm_dof_acc = (self.dof_vel[:, 0:7] - vel[:, 0:7]) / self._task_cfg["sim"]["dt"]
        self.dof_vel = vel

        # Jacobian shape: [4, 11, 6, 9] (root has no Jacobian)
        self.franka_jacobian = self.frankas.get_jacobians()
        self.franka_mass_matrix = self.frankas.get_mass_matrices(clone=False)

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_mass_matrix = self.franka_mass_matrix[
            :, 0:7, 0:7
        ]  # for Franka arm (not gripper)

        self.franka_coriolis_forces = self.frankas.get_coriolis_and_centrifugal_forces(clone=False)
        self.arm_coriolis_forces = self.franka_coriolis_forces[:, 0:7]
        
        self.franka_gravity_torque = self.frankas.get_generalized_gravity_forces(clone=False)
        


        self.claw_pos, self.claw_quat = self.frankas._claw_fixed_body.get_world_poses(clone=False)
        self.claw_pos -= self.env_pos
        claw_velocities = self.frankas._claw_fixed_body.get_velocities(clone=False)
        self.claw_linvel = claw_velocities[:, 0:3]
        self.claw_angvel = claw_velocities[:, 3:6]
       
       
        (
            self.fixed_finger_pos,
            self.fixed_finger_quat,
        ) = self.frankas._caw_fixed_finger.get_world_poses(clone=False)

        self.fixed_finger_pos -= self.env_pos
        fixed_finger_velocities = self.frankas._caw_fixed_finger.get_velocities(clone=False)
        self.fixed_finger_linvel = fixed_finger_velocities[:, 0:3]
        self.fixed_finger_angvel = fixed_finger_velocities[:, 3:6]
        self.fixed_finger_jacobian = self.franka_jacobian[:, 8, 0:6, 0:7]
        fixed_finger_forces = self.frankas._caw_fixed_finger.get_net_contact_forces(clone=False)
        # self.fixed_finger_force = fixed_finger_forces[:, 0:3]

        (
            self.claw_moving_body_pos,
            self.claw_moving_body_quat,
        ) = self.frankas._claw_moving_body.get_world_poses(clone=False)

        self.claw_moving_body_pos -= self.env_pos
        claw_moving_body_velocities = self.frankas._claw_moving_body.get_velocities(clone=False)
        self.claw_moving_body_linvel = claw_moving_body_velocities[:, 0:3]
        self.claw_moving_body_angvel = claw_moving_body_velocities[:, 3:6]
        claw_moving_body_forces = self.frankas._claw_moving_body.get_net_contact_forces(clone=False)
        # self.claw_moving_body_force = claw_moving_body_forces[:, 0:3]

        self.gripper_dof_pos = self.dof_pos[:, 7]

        # (
        #     self.fingertip_centered_pos,
        #     self.fingertip_centered_quat,
        # ) = self.frankas._fingertip_centered.get_world_poses(clone=False)
        # self.fingertip_centered_pos -= self.env_pos
        # fingertip_centered_velocities = self.frankas._fingertip_centered.get_velocities(
        #     clone=False
        # )
        # self.fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        # self.fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]
        # self.fingertip_centered_jacobian = self.franka_jacobian[:, 10, 0:6, 0:7]

        # self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) / 2
        # self.fingertip_midpoint_pos = fc.translate_along_local_z(
        #     pos=self.finger_midpoint_pos,
        #     quat=self.hand_quat,
        #     offset=self.asset_info_franka_table.franka_finger_length,
        #     device=self.device,
        # )
        # self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal

        # # TODO: Add relative velocity term (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        # self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(
        #     self.fingertip_centered_angvel,
        #     (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
        #     dim=1,
        # )

        # # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # # angular velocity of midpoint w.r.t. world is equal to sum of
        # # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        # self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal

        # self.fingertip_midpoint_jacobian = (
        #     self.left_finger_jacobian + self.right_finger_jacobian
        # ) * 0.5

        collisions_data =  self.tables.get_contact_force_data()
        self.collision_count = torch.sum(collisions_data[4][:], dim=1)

        if self.cfg_base.env.spawn_obstacle:
            obstacle_collisions = self._obstacles.get_contact_force_data()
            self.collision_count += torch.sum(obstacle_collisions[4][:], dim=1)


    def parse_controller_spec(self, add_to_stage):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {
            "num_envs",
            "jacobian_type",
            "gripper_prop_gains",
            "gripper_deriv_gains",
            "motor_ctrl_mode",
            "gain_space",
            "ik_method",
            "joint_prop_gains",
            "joint_deriv_gains",
            "do_motion_ctrl",
            "task_prop_gains",
            "task_deriv_gains",
            "do_inertial_comp",
            "motion_ctrl_axes",
            "do_force_ctrl",
            "force_ctrl_method",
            "wrench_prop_gains",
            "force_ctrl_axes",
            "use_computed_torque"
        }
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl["num_envs"] = self.num_envs
        self.cfg_ctrl["jacobian_type"] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_deriv_gains, device=self.device
        ).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type

        # configure directly gym control
        self.cfg_ctrl["motor_ctrl_mode"] = "gym"
        self.cfg_ctrl["gain_space"] = "joint"
        self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.gym_default.ik_method
        self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
            self.cfg_task.ctrl.gym_default.joint_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
            self.cfg_task.ctrl.gym_default.joint_deriv_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
            self.cfg_task.ctrl.gym_default.gripper_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
            self.cfg_task.ctrl.gym_default.gripper_deriv_gains, device=self.device
        ).repeat((self.num_envs, 1))

        if add_to_stage:
            for i in range(7):
                joint_prim = self._stage.GetPrimAtPath(
                    self.default_zero_env_path
                    + f"/franka/panda_link{i}/panda_joint{i+1}"
                )
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive.GetStiffnessAttr().Set(
                    self.cfg_ctrl["joint_prop_gains"][0, i].item() * np.pi / 180
                )
                drive.GetDampingAttr().Set(
                    self.cfg_ctrl["joint_deriv_gains"][0, i].item() * np.pi / 180
                )

            joint_prim = self._stage.GetPrimAtPath(
                self.default_zero_env_path
                + f"/franka/qbsoftclaw_body/qbsoftclaw_shaft_joint"
            )
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(
                self.cfg_ctrl["gripper_prop_gains"][0, 0].item()  * np.pi / 180
            )
            drive.GetDampingAttr().Set(
                self.cfg_ctrl["gripper_deriv_gains"][0, 0].item()  * np.pi / 180
            )


    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets or DOF torques."""

        # Get desired Jacobian
        self.fingertip_midpoint_jacobian_tf = self.fixed_finger_jacobian

        # Set PD joint pos target or joint torque
        self._set_dof_pos_target()


    def _set_dof_pos_target(self):
        """Set Franka DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fixed_finger_pos,
            fingertip_midpoint_quat=self.fixed_finger_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device,
        )
        self.frankas.set_joint_efforts(efforts=self.franka_gravity_torque)
        self.frankas.set_joint_position_targets(positions=self.ctrl_target_dof_pos)


    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        gravity = [0.0, 0.0, -gravity_mag]
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self.world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    