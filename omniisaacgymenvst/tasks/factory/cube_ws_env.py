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

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_nut_bolt.yaml.
"""

import hydra
import numpy as np
import os
import torch

from omniisaacgymenvst.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvst.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from omniisaacgymenvst.tasks.factory.cube_ws_base import CubeBase
from omni.isaac.core.materials import PreviewSurface

import omniisaacgymenvst.tasks.factory.factory_control as fc

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims import RigidPrim, RigidPrimView, XFormPrim, XFormPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import VisualSphere, VisualCylinder
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial

# from omni.kit.viewport.utility.camera_state import ViewportCameraState
# from omni.kit.viewport.utility import get_viewport_from_window_name

# import omni

from omni.physx.scripts import utils

from omniisaacgymenvst.tasks.base.rl_task import RLTask
from omniisaacgymenvst.robots.articulations.views.factory_franka_view import FactoryFrankaView

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from omni.physx.scripts import utils, physicsUtils




class CubeWS(CubeBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._get_env_yaml_params()

        super().__init__(name, sim_config, env)
    

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvNutBolt.yaml'  # relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        #asset_info_path = '../tasks/factory/yaml/factory_asset_info_nut_bolt.yaml'
        #self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        #self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['tasks']['factory']['yaml']  # strip superfluous nesting
    

    def set_up_scene(self, scene) -> None:
        self.import_franka_assets()
        self.get_cube()
        self.get_sphere()
        self.get_table()

        RLTask.set_up_scene(self, scene, replicate_physics=False)

        if self._task_cfg["env"]["use_box"]:
            self.get_box()
            self._box = RigidPrimView(prim_paths_expr="/World/envs/.*/box", 
                            name="box_view", 
                            reset_xform_properties=False,
                            contact_filter_prim_paths_expr=["/World/envs/.*/franka/panda_link{0}".format(i) for i in range(2, 8)] +
                                                            ["/World/envs/.*/franka/panda_leftfinger"] +
                                                            ["/World/envs/.*/franka/panda_rightfinger"],
                                                            max_contact_count=10000)
            scene.add(self._box)

        self.frankas = FactoryFrankaView(prim_paths_expr="/World/envs/.*/franka", name="frankas_view")
        self._cube = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", 
                                   name="cube_view", 
                                   reset_xform_properties=False)

        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)
        
        self.tables = RigidPrimView(prim_paths_expr="/World/envs/.*/table", 
                                    name="table_view",
                                    reset_xform_properties=False,
                                    contact_filter_prim_paths_expr=["/World/envs/.*/franka/panda_link{0}".format(i) for i in range(2, 8)] +
                                                                    ["/World/envs/.*/franka/panda_leftfinger"] +
                                                                    ["/World/envs/.*/franka/panda_rightfinger"],
                                                                    max_contact_count=10000)
        scene.add(self.tables)
        
        
        # Scale every cube of a factor from 1 to 4
        scales = torch.arange(1.0, self.cfg_base.env.cube_max_scale, (self.cfg_base.env.cube_max_scale - 1.0 )/ self._num_envs).to(self._device)
        # has to be (num_envs, 3)
        scales = torch.stack([scales, scales, scales], dim=1)
        self._cube.set_local_scales(scales)
        scene.add(self._cube)
        # Randomize densities between 6000 and 15000 kg/m^3
        min_dens = self.cfg_base.env.cube_min_density
        max_dens = self.cfg_base.env.cube_max_density
        densities = torch.rand((self._num_envs,), device=self._device, dtype=torch.float32) * (max_dens - min_dens) + min_dens
        self._cube.set_densities(densities)
                
        self._sphere = XFormPrimView(prim_paths_expr="/World/envs/.*/sphere", name="sphere_view")
        scene.add(self._sphere)


        if self.cfg_base.env.spawn_obstacle:
            self.get_obstacle()
            self._obstacles = RigidPrimView(prim_paths_expr="/World/envs/.*/table", 
                                    name="table_view",
                                    reset_xform_properties=False,
                                    contact_filter_prim_paths_expr=["/World/envs/.*/franka/panda_link{0}".format(i) for i in range(2, 8)] +
                                                                    ["/World/envs/.*/franka/panda_leftfinger"] +
                                                                    ["/World/envs/.*/franka/panda_rightfinger"],
                                                                    max_contact_count=10000)
            scene.add(self._obstacles)
        
        
        
        if self._cfg["test"]:
            self.get_gripper_cyl()
            self._gripper_cyl = XFormPrimView(prim_paths_expr="/World/envs/.*/gripper_cyl", name="gripper_cyl_view")
            scene.add(self._gripper_cyl)
        
        self.cube_inward_axis = torch.tensor([0, 0, -1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.cube_up_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        return
    
    def get_table(self):
        table_translation = np.array(
                [0.2 + 0.7 , 0.0, self.cfg_base.env.table_height * 0.5]
        )
        table_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        table = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_translation,
            orientation=table_orientation,
            scale=np.array(
                [
                    1.4,
                    1.4,
                    self.cfg_base.env.table_height
                ]),
            color=np.array([0.1, 0.1, 0.1]),
        )
        material = PhysicsMaterial(
            prim_path="/World/physics_material/rubber",  # path to the material prim to create
            dynamic_friction=0.8,# dynamic_friction=3.0,
            static_friction=1.1, #3.0
            restitution=0.0
        )

        table.apply_physics_material(material)
  
        self._sim_config.apply_articulation_settings(
            "table",
            get_prim_at_path(table.prim_path),
            self._sim_config.parse_actor_config("table"),
        )
        ####**** FIXED JOINT FROM WORLD TO TABLE ****####

        joint_prim = utils.createJoint(stage=get_current_stage(), 
                                       joint_type="Fixed", 
                                       from_prim=get_prim_at_path(self.default_zero_env_path), 
                                       to_prim=get_prim_at_path(table.prim_path))


    def get_cube(self):
        cube_pos = torch.tensor([0.5, 0.0, self.cfg_base.env.table_height + 0.03])

        cube = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/cube",
            name="cube",
            color=torch.tensor([0.0, 0.5, 1.0]),
            size= self.cfg_base.env.cube_size, #0.02,  # Cube size is 2x2x2cm -> 0.02*0.02*0.02 = 8e-6 m^3
            density=12500.0,  # 1000kg/m^3 -> 8g
            position=cube_pos.numpy()
        )
        material = PhysicsMaterial(
            prim_path="/World/physics_material/woo",  # path to the material prim to create
            dynamic_friction=3.0,
            static_friction=3.0,
            restitution=0.0
        )

        #cube.apply_physics_material(material)
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))

    def get_sphere(self):
        # sphere = VisualSphere(
        #     prim_path=self.default_zero_env_path + "/sphere",
        #     name="sphere",
        #     color=torch.tensor([1.0, 0.0, 0.0]),
        #     radius=0.01
        # )

        sphere = VisualCylinder(
            prim_path=self.default_zero_env_path + "/sphere",
            name="sphere",
            color=torch.tensor([1.0, 0.0, 0.0]),
            radius = self._task_cfg["env"]["goal_zone_radius"],
            height=0.01
        )
        self._sim_config.apply_articulation_settings("sphere", get_prim_at_path(sphere.prim_path), self._sim_config.parse_actor_config("sphere"))

    def get_gripper_cyl(self):
        gripper_cyl = VisualCylinder(
            prim_path=self.default_zero_env_path + "/gripper_cyl",
            name="gripper_cyl",
            color=torch.tensor([0.0, 1.0, 0.0]),
            radius=0.005,
            height=0.03
        )
        self._sim_config.apply_articulation_settings("gripper_cyl", get_prim_at_path(gripper_cyl.prim_path), self._sim_config.parse_actor_config("gripper_cyl"))
    
    def get_box(self):
        usd_path = '/workspace/omniisaacgymenvst/omniisaacgymenvst/tasks/factory/assets/small_KLT.usd'
        add_reference_to_stage(usd_path, self.default_zero_env_path + "/box")
        box_pos = torch.tensor([0.0, 0.5, self.cfg_base.env.table_height + 0.2])
        box = RigidPrim(
            prim_path=self.default_zero_env_path + "/box",
            name="box",
            position=box_pos.numpy(),
            scale=(0.5, 0.5, 0.3)
        )
        self._sim_config.apply_articulation_settings("box", get_prim_at_path(box.prim_path), self._sim_config.parse_actor_config("box"))

    def get_obstacle(self):
        barrier = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/barrier",
            name="barrier",
            translation=(-0.6, 0.0, self.cfg_base.env.table_height + 0.025),
            scale=torch.tensor([0.02, 1.1, 0.05]),
            color=torch.tensor([0.0, 0.0, 0.0]),
        )
        #self._sim_config.apply_articulation_settings("barrier", get_prim_at_path(barrier.prim_path), self._sim_config.parse_actor_config("barrier"))
        material = PhysicsMaterial(
            prim_path="/World/physics_material/rubber2",  # path to the material prim to create
            dynamic_friction=0.8,# dynamic_friction=3.0,
            static_friction=1.1, #3.0
            restitution=0.1
        )

        # barrier.apply_physics_material(material)
  
        # self._sim_config.apply_articulation_settings(
        #     "table",
        #     get_prim_at_path(table.prim_path),
        #     self._sim_config.parse_actor_config("table"),
        # )
        ####**** FIXED JOINT FROM WORLD TO TABLE ****####

        joint_prim = utils.createJoint(stage=get_current_stage(), 
                                       joint_type="Fixed", 
                                       from_prim=get_prim_at_path(self.default_zero_env_path), 
                                       to_prim=get_prim_at_path(barrier.prim_path))

    def _import_env_assets(self):
        pass

    def refresh_env_tensors(self):
        """Refresh tensors."""

        self.cube_pos, self.cube_quat = self._cube.get_world_poses(clone=False)
        self.cube_pos -= self.env_pos
        cube_velocities = self._cube.get_velocities(clone=False)
        self.cube_linvel = cube_velocities[:, 0:3]
        self.cube_angvel = cube_velocities[:, 3:6]

        # net contact force is not available yet
        # self.cube_force = ...


        # self.nut_com_pos = fc.translate_along_local_z(
        #     pos=self.nut_pos,
        #     quat=self.nut_quat,
        #     offset=self.bolt_head_heights + self.nut_heights * 0.5,
        #     device=self.device
        # )

        # self.nut_com_quat = self.nut_quat  # always equal

        # self.nut_com_linvel = self.nut_linvel + torch.cross(
        #     self.nut_angvel,
        #     (self.nut_com_pos - self.nut_pos),
        #     dim=1
        # )

    # def create_camera(self):
    #     stage = omni.usd.get_context().get_stage()
    #     self.view_port = get_viewport_from_window_name("Viewport")
    #     # Create camera
    #     self.camera_path = "/World/Camera"
    #     self.perspective_path = "/OmniverseKit_Persp"
    #     camera_prim = stage.DefinePrim(self.camera_path, "Camera")
    #     camera_prim.GetAttribute("focalLength").Set(8.5)
    #     coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
    #     if not coi_prop or not coi_prop.IsValid():
    #         camera_prim.CreateAttribute(
    #             "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
    #         ).Set(Gf.Vec3d(0, 0, -10))
    #     self.view_port.set_active_camera(self.perspective_path)