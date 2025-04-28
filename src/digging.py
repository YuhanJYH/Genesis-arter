import numpy as np

import genesis as gs
gs.init(backend=gs.gpu)


########################## create a scene ##########################
scene = gs.Scene(
    show_FPS=True,
    show_viewer=True,
    sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(3840, 2160),
        camera_pos=(0.0, -2, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=100,
    ),
    vis_options=gs.options.VisOptions(
      show_world_frame=True,
      ambient_light=(0.5, 0.5, 0.5),
    ),
    # rigid_options=gs.options.RigidOptions(
    #     enable_joint_limit=True,
    #     gravity=[0.0, 0.0, -9.81],
    #     enable_self_collision=True,
    # ),
    renderer=gs.renderers.Rasterizer(),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(
      euler = (0, 3, 0), # degrees
      ),
)

arter = scene.add_entity(
  gs.morphs.URDF(file = "/workspace/genesis/assets/urdf/arter/arter.urdf",
                 pos = (0, 0, 0.4551),
                 merge_fixed_links=False
                 ),
  visualize_contact=False,
)
# scene.build()
n_envs = 1
scene.build(n_envs=n_envs, env_spacing=(10.0, 10.0))

################# set robot config ############

swiviel_jnts = ["swivel_fl", "swivel_fr", "swivel_rl", "swivel_rr"]
swiviel_jnts_idx = [arter.get_joint(name).dof_idx_local for name in swiviel_jnts]
stabilizer_jnts = ["stabilizer_fl", "stabilizer_fr", "stabilizer_rl", "stabilizer_rr"]
stabilizer_jnts_idx = [arter.get_joint(name).dof_idx_local for name in stabilizer_jnts]
steerring_jnts = ["steering_fl", "steering_fr", "steering_rl", "steering_rr"]
steerring_jnts_idx = [arter.get_joint(name).dof_idx_local for name in steerring_jnts]
wheel_jnts = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
wheel_jnts_idx = [arter.get_joint(name).dof_idx_local for name in wheel_jnts]
claw_jnts = ["claw_dipper_rl", "claw_dipper_rr", "claw_telescope_rl", "claw_telescope_rr"]
claw_jnts_idx = [arter.get_joint(name).dof_idx_local for name in claw_jnts]
# chasis is the total of the above joints
chasis_jnts = swiviel_jnts + stabilizer_jnts + steerring_jnts + claw_jnts
chasis_jnts_idx = swiviel_jnts_idx + stabilizer_jnts_idx + steerring_jnts_idx + claw_jnts_idx

manipulator_jnts = ["cabin", "boom", "dipper", "telescope", "shovel", "roto" , "tilt"]
manipulator_jnts_idx = [arter.get_joint(name).dof_idx_local for name in manipulator_jnts]
print("manipulator_jnts_idx", manipulator_jnts_idx)

whole_jnts = manipulator_jnts + chasis_jnts + wheel_jnts
whole_jnts_idx = manipulator_jnts_idx + chasis_jnts_idx + wheel_jnts_idx

arter.set_dofs_force_range(
    lower=[int(-1e12)] * len(whole_jnts_idx),
    upper=[int(1e12)] * len(whole_jnts_idx),
    dofs_idx_local=whole_jnts_idx,
)

arter.set_dofs_kp(
    kp=[int(1e10)] * len(whole_jnts_idx),
    dofs_idx_local=whole_jnts_idx,
)

arter.set_dofs_kv(
    kv=[int(1e10)] * len(whole_jnts_idx),
    dofs_idx_local=whole_jnts_idx,
)
########################## build ##########################
# target_quat = np.tile(np.array([0, 1, 0, 0]), [n_envs, 1])  # pointing downwards
target_quat = np.tile(np.array([0, 0, 0, 1]), [n_envs, 1])  # pointing upwards
center = np.tile(np.array([2.0, 0.0, 2.0]), [n_envs, 1])

angular_speed = np.random.uniform(-10, 10, n_envs)
r = 0.5

ee_link = arter.get_link("arter/wide_shovel_bottom_link")
arter.set_dofs_position([0, 1, 0.5, 0, 0, 0, 0], manipulator_jnts_idx)
chasis_jnts_pos = np.zeros(len(chasis_jnts_idx))
chasis_jnts_pos[4:8] = [-0.2, -0.2, -0.2, -0.2]
arter.set_dofs_position(chasis_jnts_pos, chasis_jnts_idx)

for i in range(0, int(1e8)):
    # arter.control_dofs_velocity([0]*len(chasis_jnts_idx), chasis_jnts_idx)
    arter.control_dofs_velocity([0]*len(chasis_jnts_idx), chasis_jnts_idx)
    arter.control_dofs_velocity([0]*len(manipulator_jnts_idx), manipulator_jnts_idx)
    # arter.control_dofs_velocity([0]*len(wheel_jnts_idx), wheel_jnts_idx)
    
    # diff = np.random.uniform(-1,1)
    # arter.control_dofs_velocity([0, 0+diff, 0+diff, 0, 0, 0, 0], manipulator_jnts_idx)
    # arter.control_dofs_velocity([0, 1.0+diff, 0.5+diff, 0, 0, 0, 0], manipulator_jnts_idx)

#     dof_pos = np.zeros(len(whole_jnts_idx))
#     arter.set_dofs_position(dof_pos, whole_jnts_idx)
  
    # target_pos = np.zeros([n_envs, 3])
    # target_pos[:, 0] = center[:, 0] + np.cos(i / 360 * np.pi * angular_speed) * r
    # target_pos[:, 1] = center[:, 1] + np.sin(i / 360 * np.pi * angular_speed) * r
    # target_pos[:, 2] = center[:, 2]
    # target_q = np.hstack([target_pos, target_quat])

    # q = arter.inverse_kinematics(
    #     link=ee_link,
    #     pos=target_pos,
    #     quat=target_quat,
    #     rot_mask=[False, False, True],  # for demo purpose: only restrict direction of z-axis
    # )
    
    # arter.set_qpos(q)
    
    # arter.control_dofs_position(q, whole_jnts_idx)
    # arter.set_dofs_position(q)
    # scene.visualizer.update
    # arter.set_dofs_position(np.zeros(len(chasis_jnts_idx)), chasis_jnts_idx)
    # arter.set_dofs_position(np.zeros(len(manipulator_jnts)), manipulator_jnts_idx)
    scene.step()
    print("dof_poses: ", arter.get_dofs_position())
    print("base pos: ", arter.get_pos())
    print("base quat: ", arter.get_quat())
