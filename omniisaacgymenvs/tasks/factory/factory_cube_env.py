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

from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
import omniisaacgymenvs.tasks.factory.factory_control as fc

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims import RigidPrim, RigidPrimView, XFormPrim, XFormPrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import VisualSphere, VisualCylinder



from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.views.factory_franka_view import FactoryFrankaView

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from omni.physx.scripts import utils, physicsUtils




class FactoryCube(FactoryBase, FactoryABCEnv):
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
        
        RLTask.set_up_scene(self, scene, replicate_physics=False)

        self.frankas = FactoryFrankaView(prim_paths_expr="/World/envs/.*/franka", name="frankas_view")
        self._cube = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", 
                                   name="cube_view", 
                                   reset_xform_properties=False)
        
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)
        scene.add(self._cube)
        
        if self._cfg["test"]:
            self.get_gripper_cyl()
            self.get_sphere()
            self._sphere = XFormPrimView(prim_paths_expr="/World/envs/.*/sphere", name="sphere_view")
            self._gripper_cyl = XFormPrimView(prim_paths_expr="/World/envs/.*/gripper_cyl", name="gripper_cyl_view")
            scene.add(self._sphere)
            scene.add(self._gripper_cyl)
        
        self.cube_inward_axis = torch.tensor([0, 0, -1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.cube_up_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        return
    
    def get_cube(self):
        cube_pos = torch.tensor([0.5, 0.0, self.cfg_base.env.table_height + 0.03])

        cube = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/cube",
            name="cube",
            color=torch.tensor([0.0, 0.5, 1.0]),
            size=0.02,  # Cube size is 2x2x2cm
            density=100.0,
            position=cube_pos.numpy()
        )
        self._sim_config.apply_articulation_settings("cube", get_prim_at_path(cube.prim_path), self._sim_config.parse_actor_config("cube"))

    def get_sphere(self):
        sphere = VisualSphere(
            prim_path=self.default_zero_env_path + "/sphere",
            name="sphere",
            color=torch.tensor([1.0, 0.0, 0.0]),
            radius=0.01
        )
        self._sim_config.apply_articulation_settings("sphere", get_prim_at_path(sphere.prim_path), self._sim_config.parse_actor_config("sphere"))

    def get_gripper_cyl(self):
        gripper_cyl = VisualCylinder(
            prim_path=self.default_zero_env_path + "/gripper_cyl",
            name="gripper_cyl",
            color=torch.tensor([0.0, 1.0, 1.0]),
            radius=0.01,
            height=0.05
        )
        self._sim_config.apply_articulation_settings("gripper_cyl", get_prim_at_path(gripper_cyl.prim_path), self._sim_config.parse_actor_config("gripper_cyl"))
    
    def _import_env_assets(self):
        """Set nut and bolt asset options. Import assets."""

        self.nut_heights = []
        self.nut_widths_max = []
        self.bolt_widths = []
        self.bolt_head_heights = []
        self.bolt_shank_lengths = []
        self.thread_pitches = []

        assets_root_path = get_assets_root_path()
        
        for i in range(0, self._num_envs):

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_nut_bolt[subassembly])

            nut_translation = torch.tensor([0.0, self.cfg_env.env.nut_lateral_offset, self.cfg_base.env.table_height], device=self._device)
            nut_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            nut_height = self.asset_info_nut_bolt[subassembly][components[0]]['height']
            nut_width_max = self.asset_info_nut_bolt[subassembly][components[0]]['width_max']
            self.nut_heights.append(nut_height)
            self.nut_widths_max.append(nut_width_max)

            nut_file = assets_root_path + self.asset_info_nut_bolt[subassembly][components[0]]['usd_path']

            add_reference_to_stage(nut_file, f"/World/envs/env_{i}" + "/nut")
            nut_prim = XFormPrim(
                prim_path=f"/World/envs/env_{i}" + "/nut",
                translation=nut_translation,
                orientation=nut_orientation,
            )

            physicsUtils.add_physics_material_to_prim(
                self._stage, 
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + f"/nut/factory_{components[0][0:-6]}/collisions/mesh_0"), 
                self.nutboltPhysicsMaterialPath
            )

            bolt_translation = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self._device)
            bolt_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            bolt_width = self.asset_info_nut_bolt[subassembly][components[1]]['width']

            bolt_head_height = self.asset_info_nut_bolt[subassembly][components[1]]['head_height']
            bolt_shank_length = self.asset_info_nut_bolt[subassembly][components[1]]['shank_length']
            self.bolt_widths.append(bolt_width)
            self.bolt_head_heights.append(bolt_head_height)
            self.bolt_shank_lengths.append(bolt_shank_length)

            bolt_file = assets_root_path + self.asset_info_nut_bolt[subassembly][components[1]]['usd_path']
            add_reference_to_stage(bolt_file, f"/World/envs/env_{i}" + "/bolt")
            XFormPrim(
                prim_path=f"/World/envs/env_{i}" + "/bolt",
                translation=bolt_translation,
                orientation=bolt_orientation,
            )

            physicsUtils.add_physics_material_to_prim(
                self._stage, 
                self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + f"/bolt/factory_{components[1][0:-6]}/collisions/mesh_0"), 
                self.nutboltPhysicsMaterialPath
            )

            thread_pitch = self.asset_info_nut_bolt[subassembly]['thread_pitch']
            self.thread_pitches.append(thread_pitch)

        # For computing body COM pos
        self.nut_heights = torch.tensor(self.nut_heights, device=self._device).unsqueeze(-1)
        self.bolt_head_heights = torch.tensor(self.bolt_head_heights, device=self._device).unsqueeze(-1)

        # For setting initial state
        self.nut_widths_max = torch.tensor(self.nut_widths_max, device=self._device).unsqueeze(-1)
        self.bolt_shank_lengths = torch.tensor(self.bolt_shank_lengths, device=self._device).unsqueeze(-1)

        # For defining success or failure
        self.bolt_widths = torch.tensor(self.bolt_widths, device=self._device).unsqueeze(-1)
        self.thread_pitches = torch.tensor(self.thread_pitches, device=self._device).unsqueeze(-1)
    

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

