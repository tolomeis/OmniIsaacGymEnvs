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


def import_tasks():
    from omniisaacgymenvst.tasks.allegro_hand import AllegroHandTask
    from omniisaacgymenvst.tasks.ant import AntLocomotionTask
    from omniisaacgymenvst.tasks.anymal import AnymalTask
    from omniisaacgymenvst.tasks.anymal_terrain import AnymalTerrainTask
    from omniisaacgymenvst.tasks.ball_balance import BallBalanceTask
    from omniisaacgymenvst.tasks.cartpole import CartpoleTask
    from omniisaacgymenvst.tasks.cartpole_camera import CartpoleCameraTask
    from omniisaacgymenvst.tasks.crazyflie import CrazyflieTask
    from omniisaacgymenvst.tasks.factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
    from omniisaacgymenvst.tasks.factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
    from omniisaacgymenvst.tasks.factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
    from omniisaacgymenvst.tasks.factory.factory_task_cube import FactoryCubeTask
    from omniisaacgymenvst.tasks.factory.cube_task import CubeTask
    from omniisaacgymenvst.tasks.factory.softclaw_cube_task import SoftclawCubeTask
    from omniisaacgymenvst.tasks.factory.cube_mobile_task import CubeMobileTask

    from omniisaacgymenvst.tasks.franka_cabinet import FrankaCabinetTask
    from omniisaacgymenvst.tasks.franka_deformable import FrankaDeformableTask
    from omniisaacgymenvst.tasks.humanoid import HumanoidLocomotionTask
    from omniisaacgymenvst.tasks.ingenuity import IngenuityTask
    from omniisaacgymenvst.tasks.quadcopter import QuadcopterTask
    from omniisaacgymenvst.tasks.shadow_hand import ShadowHandTask

    from omniisaacgymenvst.tasks.warp.ant import AntLocomotionTask as AntLocomotionTaskWarp
    from omniisaacgymenvst.tasks.warp.cartpole import CartpoleTask as CartpoleTaskWarp
    from omniisaacgymenvst.tasks.warp.humanoid import HumanoidLocomotionTask as HumanoidLocomotionTaskWarp

    # Mappings from strings to environments
    task_map = {
        "AllegroHand": AllegroHandTask,
        "Ant": AntLocomotionTask,
        "Anymal": AnymalTask,
        "AnymalTerrain": AnymalTerrainTask,
        "BallBalance": BallBalanceTask,
        "Cartpole": CartpoleTask,
        "CartpoleCamera": CartpoleCameraTask,
        "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
        "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
        "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
        "FactoryCube": FactoryCubeTask,
        "CubeTask": CubeTask,
        "SoftclawCubeTask": SoftclawCubeTask,
        "FrankaCabinet": FrankaCabinetTask,
        "FrankaDeformable": FrankaDeformableTask,
        "Humanoid": HumanoidLocomotionTask,
        "Ingenuity": IngenuityTask,
        "Quadcopter": QuadcopterTask,
        "Crazyflie": CrazyflieTask,
        "ShadowHand": ShadowHandTask,
        "ShadowHandOpenAI_FF": ShadowHandTask,
        "ShadowHandOpenAI_LSTM": ShadowHandTask,
        "CubeMobileTask": CubeMobileTask,
    }

    task_map_warp = {
        "Cartpole": CartpoleTaskWarp,
        "Ant":AntLocomotionTaskWarp,
        "Humanoid": HumanoidLocomotionTaskWarp
    }

    return task_map, task_map_warp


def initialize_task(config, env, init_sim=True):
    from omniisaacgymenvst.utils.config_utils.sim_config import SimConfig

    sim_config = SimConfig(config)
    task_map, task_map_warp = import_tasks()

    cfg = sim_config.config
    if cfg["warp"]:
        task_map = task_map_warp

    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    backend = "warp" if cfg["warp"] else "torch"

    rendering_dt = sim_config.get_physics_params()["rendering_dt"]

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend=backend,
        init_sim=init_sim,
        rendering_dt=rendering_dt,
    )

    return task
