from omni.isaac.kit import SimulationApp
import os
import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    simulation_app = SimulationApp({"headless": False, "enable_livestream": True, "physics_gpu": 0}, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.kit')

    from omniisaacgymenvst.utils.config_utils.sim_config import SimConfig
    from omniisaacgymenvst.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

    config = omegaconf_to_dict(cfg)
    sim_config = SimConfig(config)
    backend = "warp" if cfg["warp"] else "torch"
    rendering_dt = sim_config.get_physics_params()["rendering_dt"]
    device='cuda:0'
    device='cpu'

    from omni.isaac.core import World

    # from omni.isaac.manipulators import SingleManipulator
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.types import ArticulationAction
    # from omni.isaac.core.robots.robot import Robot
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import numpy as np
    from omni.isaac.core.objects import GroundPlane

    import omni.isaac.core.utils.prims as prim_utils
    from pxr import PhysxSchema
    from omni.isaac.core.materials import PhysicsMaterial

    from omniisaacgymenvst.robots.articulations.factory_franka_mobile import FactoryFrankaMobile
    from omniisaacgymenvst.robots.articulations.views.factory_franka_mobile_view import MobileFrankaView
    from omniisaacgymenvst.robots.articulations.anymal import Anymal
    from omniisaacgymenvst.robots.articulations.views.anymal_view import AnymalView
    my_world = World(stage_units_in_meters=1.0, rendering_dt=rendering_dt, backend=backend, sim_params=sim_config.get_physics_params(), device=device)
    # my_world = World(stage_units_in_meters=1.0) 

    # asset_path = "/home/darko/isaac-ws/OmniIsaacGymEnvs/assets/Collected_factory_franka_instanceable/franka_softclaw_inrt.usd"
    # asset_path = "/home/darko/isaac-ws/OmniIsaacGymEnvs/assets/darko_base/flattenedfrankabase3.usd"
    # odyn_path = "/home/darko/isaac-ws/models/Collected_o3dyn/o3dyn.usd"


    # Using classes
    prim_path = "/World/envs/envs0/franka_mobile/darko_base_link"
    robot = FactoryFrankaMobile(prim_path="/World/envs/envs0/franka_mobile", name="franka_mobile",
                                orientation=[0.0, 0.0, 0.7071, -0.7071])
    
    robot_view = MobileFrankaView(prim_paths_expr=prim_path, omniwheels_mass=[1.0, 1.0, 1.0, 1.0])

    # prim_path = "/World/envs/envs0/franka_mobile"
    # robot = FactoryFrankaMobile(prim_path="/World/envs/envs0/franka_mobile", name="franka_mobile",
    #                             orientation=[0.0, 0.0, 0.7071, -0.7071])
    
    # robot_view = MobileFrankaView(prim_paths_expr=prim_path, omniwheels_mass=[1.0, 1.0, 1.0, 1.0])

    my_world.scene.add(robot_view)

    # prim_path = "/World/envs/envs0/o3dyn"
    # add_reference_to_stage(usd_path=odyn_path, prim_path=prim_path)
    # robot_view = ArticulationView(prim_paths_expr=prim_path) #, translations=[0,0,0.5])
    # my_world.scene.add(robot_view)

    plane = GroundPlane(prim_path="/World/GroundPlane", z_position=-1)


    light_1 = prim_utils.create_prim(
        "/World/Light_1",
        "DistantLight",
        orientation=torch.tensor([0.65,0.27, 0.27, 0.65]).to(device),
    )
    # from pxr import UsdLux.LightAPI
    # PhysxSchema.LightAPI(get_prim_at_path("/World/Light_1")).CreateTemperatureAttr().Set(3000)


    my_world.reset()


    i = 0

    while simulation_app.is_running():

        my_world.step(render=True)
        # print(robot_view.get_world_poses())
        if i == 300:
            poses = torch.tensor([0.0, 0.0, 1.2], dtype=torch.float32).to(device)
            print("************************ step 300")  
            print(robot_view.body_names)
            orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).to(device)
            robot_view.set_world_poses(poses, orientations, indices=torch.tensor([0]).to(device))

        if i >= 300:
            input()
        
        # print(robot_view.get_jacobian_shape())
        # print(robot_view.body_names)
        # print(robot_view.dof_names)
        # positions = np.tile(np.array([20.0]), (1, 1)) # 1 env, 1 joint4
        
        wheel_idx = [robot_view.get_dof_index("wheel_LF_joint"), robot_view.get_dof_index("wheel_RF_joint"), 
                    robot_view.get_dof_index("wheel_LH_joint"), robot_view.get_dof_index("wheel_RH_joint")]

        panda_arm_joint_idx = [robot_view.get_dof_index("panda_joint1"), robot_view.get_dof_index("panda_joint2"), 
                robot_view.get_dof_index("panda_joint3"), robot_view.get_dof_index("panda_joint4"), 
                robot_view.get_dof_index("panda_joint5"), robot_view.get_dof_index("panda_joint6"), 
                robot_view.get_dof_index("panda_joint7")]
        
        # print(robot_view.body_names)

        if i == 600:
            poses = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32).to(device)
            print("************************ step 300")
            orientations = torch.tensor([0.7071, 0.0, 0.0, -0.7071], dtype=torch.float32).to(device)
            robot_view.set_world_poses(poses, orientations, indices=torch.tensor([0]).to(device))
        
        """
        if i == 500:
            print("************************ step 500")
            poses = torch.tensor([0.0, 0.0, 1.2], dtype=torch.float32).to(device)

            orientations = torch.tensor([0.0, 0.0, 0.7071, -0.7071], dtype=torch.float32).to(device)
            robot_view.set_world_poses(poses, orientations, indices=torch.tensor([0]).to(device))

        if i >= 800:
            print("driving")
            # robot_view.set_joint_velocity_targets(torch.tensor([-1.0, 1.0, 
            #                                     1.0, -1.0]).to(device), indices=torch.tensor([0]).to(device), joint_indices=joint_idx)
            ndof = robot_view.num_dof
            andof = 11
            # robot_view.set_velocities(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device), 
            #                                       indices=torch.tensor([0]).to(device))
            # current_pos, current_or = robot_view.get_world_poses()
            # current_pos[0,2] += 0.3
            # robot_view.set_world_poses(current_pos, current_or, indices=torch.tensor([0]).to(device))

            joint_pos_zeros = torch.zeros(andof).to(device)

            #arm random movements
            robot_view.set_joint_position_targets(6.28*torch.rand(7).to(device) - 3.14, 
                                                  indices=torch.tensor([0]).to(device),
                                                  joint_indices=torch.tensor(panda_arm_joint_idx).to(device))
            
            robot_view.set_joint_velocity_targets(torch.zeros(1).to(device), 
                                                  indices=torch.tensor([0]).to(device),
                                                  joint_indices=torch.tensor(panda_arm_joint_idx[5]).to(device))
            
            # Move Wheels
            robot_view.set_joint_velocity_targets(torch.tensor([-1.0, 1.0, 
                                                1.0, -1.0]).to(device), indices=torch.tensor([0]).to(device), 
                                                joint_indices=wheel_idx)
                                       



        PRINT_INFO=False
        if PRINT_INFO:
            print("position_iteration_count: ", robot_view.get_solver_position_iteration_counts())
            print("velocity_iteration_count: ", robot_view.get_solver_velocity_iteration_counts())
            print("stabilization threshold: ", robot_view.get_stabilization_thresholds())
            print(robot_view.get_sleep_thresholds())
            print(robot_view.get_enabled_self_collisions())
            """
        i += 1.0

    simulation_app.close()

"""
(10, 6, 8)
['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'qbsoftclaw_body', 'qbsoftclaw_shaft_link', 'fixed_finger_pad']
['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'qbsoftclaw_shaft_joint']
None
"""

if __name__ == "__main__":
    parse_hydra_configs()
