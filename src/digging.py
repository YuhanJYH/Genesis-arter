import numpy as np
import torch

# motion planning
import pybullet
import pybullet_data
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import spatialmath as sm
import matplotlib.pyplot as plt
import math

# local imports
from assets import friction_coefficients
import helpers
import arter

import genesis as gs
gs.init(backend=gs.gpu)

N_ENVS = 10
START_SIM = True
ENV_SPACING = 30.0
INIT_POS = (0.0, 0.0, 6.0)
INIT_QUAT = (0.0, 0.0, 0.0, 1.0)

########################## create a scene ##########################
scene = gs.Scene(
    show_FPS=True,
    show_viewer=False,
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(3840, 2160),
        camera_pos=(0.0, -5, 7),
        camera_lookat=(0.0, 0.0, 5),
        camera_fov=40,
        max_FPS=100,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        ambient_light=(0.5, 0.5, 0.5),
    ),
    rigid_options=gs.options.RigidOptions(
        enable_joint_limit=True,
        gravity=[0.0, 0.0, -9.81],
        enable_self_collision=True,
    ),
    renderer=gs.renderers.Rasterizer(),
)

cam = scene.add_camera(
  res= (640, 480),
  pos=(0, 20, 20),
  lookat=(0, 0, 5),
  fov=40,
  GUI=True,
)


########################## entities ##########################
platform = scene.add_entity(
    gs.morphs.Box(
        size=(10, 10, 5),
        pos=(0, 0, 2.5),
        fixed=True,
    ),
)
        
arter_entity = scene.add_entity(
    gs.morphs.URDF(
        file=arter.URDF_PATH, 
        pos=INIT_POS, 
        quat=INIT_QUAT,
        fixed=False,
        merge_fixed_links=False  # 0.4551
    ),
    visualize_contact=False,    
)
############################ genesis idx ##########################
swiviel_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.swiviel_jnts]
stabilizer_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.stabilizer_jnts]
steerring_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.steerring_jnts]
wheel_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.wheel_jnts]
wheel_vel_mask = np.array([1, -1, 1, -1])
wheel_link_names = ["arter/wheel_fl_link", "arter/wheel_fr_link", "arter/wheel_rl_link", "arter/wheel_rr_link"]
claw_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.claw_jnts]
chasis_jnts_idx = swiviel_jnts_idx + stabilizer_jnts_idx + claw_jnts_idx
manipulator_jnts_idx = [arter_entity.get_joint(name).dof_idx_local for name in arter.manipulator_jnts]
whole_jnts_idx = manipulator_jnts_idx + chasis_jnts_idx + wheel_jnts_idx + steerring_jnts_idx

print('friction wheel:',arter_entity.get_link("arter/wheel_fl_link").geoms[0].friction)

# Set the friction coefficient for the wheel links
for wheel_link_name in wheel_link_names:
    wheel_link = arter_entity.get_link(wheel_link_name)
    wheel_link.set_friction(friction_coefficients["road"]["dry_asphalt_concrete"]["min"])  # Set to min value
    
print('friction wheel:',arter_entity.get_link("arter/wheel_fl_link").geoms[0].friction)

########################### set motion planner ##########################
# pybullet.connect(pybullet.DIRECT)  # Connect in GUI-less mode
# pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
# arter_bullet = pybullet.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=False)
# num_joints = pybullet.getNumJoints(arter_bullet)
# joint_names = {}
# link_names = {}
# for i in range(num_joints):
#     joint_info = pybullet.getJointInfo(arter_bullet, i)
#     joint_name = joint_info[1].decode("utf-8")
#     link_name = joint_info[12].decode("utf-8")  # Child link name
#     joint_names[i] = joint_name
#     link_names[i] = link_name

# print("Joint Names:", joint_names)
# print("Link Names:", link_names)

# bullet_manipulator_idxs = []
# for i in range(len(manipulator_jnts_idx)):
#     bullet_manipulator_idxs.append(list(joint_names.keys())[list(joint_names.values()).index(manipulator_jnts[i])])
# print("bullet_manipulator_idxs: ", bullet_manipulator_idxs)

# num_active_joints = len(bullet_manipulator_idxs)
# space = ob.RealVectorStateSpace(num_active_joints)
# bounds = ob.RealVectorBounds(num_active_joints)

# for i, joint_index in enumerate(bullet_manipulator_idxs):
#     joint_info = pybullet.getJointInfo(arter_bullet, joint_index)
#     lower_limit = joint_info[8]
#     upper_limit = joint_info[9]
#     bounds.setLow(float(lower_limit), int(i))  # Explicitly cast i to int
#     bounds.setHigh(float(upper_limit), int(i))  # Explicitly cast i to int


########################## build ##########################

if START_SIM:
    scene.build(n_envs=N_ENVS, env_spacing=(ENV_SPACING, ENV_SPACING))
    
    ########################### set init state ##################
    manipulator_init_pos = [0, 1, 0.5, 0, 0, 0, 0]
    manipulator_init_pos = np.tile(manipulator_init_pos, (N_ENVS, 1))
    arter_entity.set_dofs_position(manipulator_init_pos, manipulator_jnts_idx)

    chasis_init_pos = np.zeros(len(chasis_jnts_idx))
    chasis_init_pos[4:8] = [-0.2, -0.2, -0.2, -0.2]
    chasis_init_pos = np.tile(chasis_init_pos, (N_ENVS, 1))
    arter_entity.set_dofs_position(chasis_init_pos, chasis_jnts_idx)

    ################# set robot controller ##################
    arter_entity.set_dofs_force_range(
        lower=[int(-1e12)] * len(whole_jnts_idx),
        upper=[int(1e12)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    arter_entity.set_dofs_kp(
        kp=[int(1e10)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    arter_entity.set_dofs_kv(
        kv=[int(1e10)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    ################## run sim ##################
    cam.start_recording()
    for i in range(0, int(1e4)):
        chasis_vel = np.zeros(len(chasis_jnts_idx))
        chasis_vel = np.tile(chasis_vel, (N_ENVS, 1))
        manipulator_vel = np.zeros(len(manipulator_jnts_idx))
        manipulator_vel = np.tile(manipulator_vel, (N_ENVS, 1))

        wheel_vel = np.array([3 * np.pi] * 4)
        wheel_vel = wheel_vel_mask * wheel_vel
        wheel_vel = np.tile(wheel_vel, (N_ENVS, 1))
        
        steer_vel = np.array([0, 0, 0, 0])
        steer_vel = np.tile(steer_vel, (N_ENVS, 1))

        arter_entity.control_dofs_velocity(chasis_vel, chasis_jnts_idx)
        arter_entity.control_dofs_velocity(manipulator_vel, manipulator_jnts_idx)
        arter_entity.control_dofs_velocity(wheel_vel, wheel_jnts_idx)
        arter_entity.control_dofs_velocity(steer_vel, steerring_jnts_idx)
        
        scene.step()
        cam.render(depth=False)
        
        base_pos = arter_entity.get_pos()
        base_quat = arter_entity.get_quat()
        
        # reset to init at tipping
        for i in range(N_ENVS):
            if arter_entity.get_pos()[i][2] < 5.5:
                print("arter is dropped")
                for j in range(len(base_pos[i])):
                    base_pos[i][j] = INIT_POS[j]
                arter_entity.set_pos(base_pos)
                for j in range(len(base_quat[i])):
                    base_quat[i][j] = INIT_QUAT[j]
                arter_entity.set_quat(base_quat)
                
    cam.stop_recording(save_to_filename="arter_drop.mp4", fps=60)