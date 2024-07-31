# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvst.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema


class FrankaSoftclaw(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "franka",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        # self._usd_path = "/home/darko/isaac-ws/OmniIsaacGymEnvs/assets/Collected_factory_franka/factory_franka.usd"
        self._usd_path = "/home/darko/isaac-ws/OmniIsaacGymEnvs/assets/Collected_factory_franka_instanceable/franka_softclaw_inrt.usd"
        print(self._usd_path)
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "panda_link0/panda_joint1",
            "panda_link1/panda_joint2",
            "panda_link2/panda_joint3",
            "panda_link3/panda_joint4",
            "panda_link4/panda_joint5",
            "panda_link5/panda_joint6",
            "panda_link6/panda_joint7",
            "qbsoftclaw_body/qbsoftclaw_shaft_joint",
        ]

        drive_type = ["angular"] * 8
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.0]
        stiffness = [40 * np.pi / 180] * 7 + [30 * np.pi / 180]
        damping = [80 * np.pi / 180] * 7 + [20]

        max_force = [x*0.90 for x in [82.65, 82.65, 82.65, 82.65, 11.4, 11.4, 11.4, 6.5]]

        max_velocity = [math.degrees(x*0.90) for x in [2.06, 2.06, 2.06, 2.06, 2.48, 2.48, 2.48]] + [340]
        # TODO: check limits in datasheet
        
        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )
