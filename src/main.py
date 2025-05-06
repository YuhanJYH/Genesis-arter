import genesis as gs

gs.init()
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
import math
import pinocchio as pin
import arter_pin

# import rsl_rl

# local imports
from assets import wheel_friction_coefficients
import helpers
import arter
import generate_platform

RECORD = False
DEBUG_MODE = False

N_ENVS = 1
ENV_SPACING = 30.0

INIT_POS = (0.0, 0.0, 6.0)
INIT_QUAT = (0.0, 0.0, 0.0, 1.0)
INIT_EULER = (0,0,90)
########################## create a scene ##########################
scene = gs.Scene(
    renderer=gs.renderers.Rasterizer(),  # raytracer need luisa renderer
    show_FPS=True,
    show_viewer=not RECORD,
    # overarch options
    sim_options=gs.options.SimOptions(
        dt=1e-2,  # default 1e-2
        gravity=(0, 0, -9.8),  # default (0, 0, -9.81)
    ),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_joint_limit=True,
        enable_self_collision=True,
        enable_adjacent_collision=False,
        max_collision_pairs=50,  # default 100, save some resource
    ),
    viewer_options=gs.options.ViewerOptions(
        # res=(3840, 2160),
        # refresh_rate=30,
        camera_pos=(0.0, -5, 7),
        camera_lookat=(0.0, 0.0, 5),
        camera_fov=40,
        max_FPS=None,  # full speed
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=True,
        link_frame_size=0.3,
        show_cameras=True,
        shadow=False,  # to save some resource
        plane_reflection=False,
        ambient_light=(1.0, 1.0, 1.0),  # color of the ambient light
        visualize_mpm_boundary=False,
        visualize_sph_boundary=False,
        visualize_pbd_boundary=False,
        rendered_envs_idx=[0],  # only render the first env
    ),
)


# visualizer debug info
print(scene.sim)
print(scene.rigid_options)
print(scene.viewer_options)
print(scene.vis_options)

if RECORD:
    cam = scene.add_camera(
        res=(2160, 1440),
        pos=(0, 200, 30),
        lookat=(0, 0, 5),
        fov=40,
        GUI=False,
    )

########################## entities ##########################
# generate platform
platform_file = "/workspace/tmp/platform.stl"
generate_platform.create_sloped_platform(top_width=10, height=5, slope_angle_deg=60, output_path=platform_file)
platform_morph = gs.morphs.Mesh(
    file=platform_file,
    decimate=False,
    convexify=False,
    quality=False,
    collision=True,
    visualization=True,
    pos=(0, 0, 0),
    fixed=True,
)

platform_ett = scene.add_entity(
    platform_morph,
    visualize_contact=False,
)

arter_ett = scene.add_entity(
    gs.morphs.URDF(
        file=arter.URDF_PATH,
        pos=INIT_POS,
        quat=INIT_QUAT,
        # euler=INIT_EULER,
        fixed=False,
        convexify=False,
        visualization=True,
        collision=True,
        requires_jac_and_IK=False,
        prioritize_urdf_material=False,
        merge_fixed_links=True,
        links_to_keep=["arter/wide_shovel_base_link", "arter/wide_shovel_bottom_link"],
    ),
    visualize_contact=False,
)

############################ genesis idx ##########################
swivel_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.swiviel_jnts]
stabilizer_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.stabilizer_jnts]
steerring_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.steerring_jnts]
wheel_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.wheel_jnts]
claw_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.claw_jnts]
chasis_jnts_idx = swivel_jnts_idx + stabilizer_jnts_idx + claw_jnts_idx
manipulator_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.manipulator_jnts]
whole_jnts_idx = manipulator_jnts_idx + chasis_jnts_idx + wheel_jnts_idx + steerring_jnts_idx

print("genesis chasis:", chasis_jnts_idx)
print("pin chasis: ", arter_pin.chasis_jnts_idx)

print("genesis manipulator: ", manipulator_jnts_idx)
print("pin manipulator:", arter_pin.manipulator_jnts_idx)


######################### friction & contact #################
print("friction wheel:", arter_ett.get_link("arter/wheel_fl_link").geoms[0].friction)
# Set the friction coefficient for the wheel link
# for wheel_link_name in arter.wheel_link_names:
#     wheel_link = arter_ett.get_link(wheel_link_name)
#     # wheel_link.set_friction(friction_coefficients["road"]["dry_asphalt_concrete"]["max"])  # Set to min value
#     wheel_link.set_friction(10.0)  # Set to min value

# print("friction wheel:", arter_ett.get_link("arter/wheel_fl_link").geoms[0].friction)

########################## build ##########################

scene.build(n_envs=N_ENVS, env_spacing=(ENV_SPACING, ENV_SPACING))

if DEBUG_MODE:
    import IPython
    IPython.embed()  # start the interactive shell

if not DEBUG_MODE:
    ########################### set init state ##################
    manipulator_init_pos = [0, 1, 0.5, 0, 0, 0, 0]  
    manipulator_init_pos = np.array(manipulator_init_pos)
    manipulator_init_pos = np.tile(manipulator_init_pos, (N_ENVS, 1))
    random_mask = np.random.uniform(low=-0.2, high=0.2, size=(N_ENVS, 7))
    manipulator_init_pos = manipulator_init_pos + random_mask
    arter_ett.set_dofs_position(manipulator_init_pos, manipulator_jnts_idx)
    print("randomized manipulator init pos: ", manipulator_init_pos)

    chasis_init_pos = np.zeros(len(chasis_jnts_idx))
    chasis_init_pos[4:8] = [-0.3, -0.3, -0.3, -0.3]
    chasis_init_pos = np.tile(chasis_init_pos, (N_ENVS, 1))
    rand = [np.random.uniform(low=-0.2, high=0.0)] * 4
    random_mask = np.array(rand)
    random_mask = np.tile(random_mask, (N_ENVS, 1))
    # chasis_init_pos[:, 4:8] = chasis_init_pos[:, 4:8] + random_mask

    arter_ett.set_dofs_position(chasis_init_pos, chasis_jnts_idx)

    ################# set robot controller ##################
    arter_ett.set_dofs_force_range(
        lower=[int(-1e12)] * len(whole_jnts_idx),
        upper=[int(1e12)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    arter_ett.set_dofs_kp(
        kp=[int(1e10)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    arter_ett.set_dofs_kv(
        kv=[int(1e10)] * len(whole_jnts_idx),
        dofs_idx_local=whole_jnts_idx,
    )

    ################## run sim ##################
    iter_num = 0
    if RECORD:
        cam.start_recording()
        iter_num = int(3 * 1e3)
    else:
        iter_num = int(1e8)

    pin_q = arter_pin.q
    pin_model = arter_pin.model
    pin_data = arter_pin.data
    for i in range(0, iter_num):
        # -------------- observe current state ---------------------------
        # read joint states
        q_obs = arter_ett.get_dofs_position()        
        # update pinocchio model
        mnp_jnts_pos = [q_obs[0][id] for id in manipulator_jnts_idx]
        arter_pin.set_mnp_jnts_pos(mnp_jnts_pos, pin_q)
        stab_jnts_pos = [q_obs[0][id] for id in stabilizer_jnts_idx]
        arter_pin.set_stabilizer_jnts_pos(stab_jnts_pos, pin_q)
        swi_jnts_pos = [q_obs[0][id] for id in swivel_jnts_idx]
        arter_pin.set_swivel_jnts_pos(swi_jnts_pos, pin_q)
        steer_jnts_pos = [q_obs[0][id] for id in steerring_jnts_idx]
        arter_pin.set_steering_jnts_pos(steer_jnts_pos, pin_q)
        pin.forwardKinematics(pin_model,pin_data,pin_q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        tip_pos = pin_data.oMf[arter_pin.tip_link_id]
        tip_pos_gs = arter_ett.get_link("arter/wide_shovel_bottom_link").get_pos()

        base_pos_gs = arter_ett.get_pos()
        base_quat_gs = arter_ett.get_quat()
        
        print(f"pin tip pos: {tip_pos}")
        print(f"gs gloabl tip pos: {tip_pos_gs}")
        print(f"gs relativ tip pos: {tip_pos_gs[0] - base_pos_gs[0]}")
        # print(f"tip pos relative to base: {tip_pos_relativ}")
        com = pin.centerOfMass(pin_model, pin_data)
        print(f"center of mass: {com}")
            
        # IMU judge
         # reset to init at tipping
        for i in range(N_ENVS):
            # if arter_entity.get_pos()[i][2] < 5.2 or arter_entity.get_pos()[i][2] > 6.5:
            env_quat = base_quat_gs[i]
            env_quat = env_quat.cpu()
            env_quat = np.array(env_quat)
            # print("env_quat: ", env_quat)
            # general tilt angle > 30 degrees
            r1 = R.from_quat(env_quat)
            r_init = R.from_quat(INIT_QUAT)
            r_diff = r1 * r_init.inv()
            rot_vector = r_diff.as_rotvec()
            diff_angle = np.linalg.norm(rot_vector)
            diff_angle = math.degrees(diff_angle)
            print("tilt angle ", i, ": ", diff_angle)

            # update pinocchio model
            # manipulator state

            if diff_angle > 40 or base_pos_gs[i][2] < 5.2 or base_pos_gs[i][2] > 6.5:
                print("arter is tipping, reset to init")
                # reset to init
                for j in range(len(base_pos_gs[i])):
                    base_pos_gs[i][j] = INIT_POS[j]
                arter_ett.set_pos(base_pos_gs)
                for j in range(len(base_quat_gs[i])):
                    base_quat_gs[i][j] = INIT_QUAT[j]
                arter_ett.set_quat(base_quat_gs)

                # give a new manipulatopr position at new init
                env_manipulator_init_pos = [0, 1, 0.5, 0, 0, 0, 0]
                env_manipulator_init_pos = np.array(env_manipulator_init_pos)
                rand_mask = np.random.uniform(low=-0.5, high=0.5, size=(1, 7))
                manipulator_init_pos[i] = env_manipulator_init_pos + rand_mask
                # update gs entity
                arter_ett.set_dofs_position(manipulator_init_pos, manipulator_jnts_idx)
                print("randomized manipulator init pos: ", manipulator_init_pos)
        
        #--------------- give control signal --------------------------
        chasis_vel = np.zeros(len(chasis_jnts_idx))
        chasis_vel = np.tile(chasis_vel, (N_ENVS, 1))
        manipulator_vel = np.zeros(len(manipulator_jnts_idx))
        manipulator_vel = np.tile(manipulator_vel, (N_ENVS, 1))

        wheel_vel = np.array([0.1 * np.pi] * 4)
        wheel_vel = arter.wheel_vel_mask * wheel_vel
        wheel_vel = np.tile(wheel_vel, (N_ENVS, 1))

        steer_vel = np.array([0, 0, 0, 0])
        steer_vel = np.tile(steer_vel, (N_ENVS, 1))

        arter_ett.control_dofs_velocity(chasis_vel, chasis_jnts_idx)
        arter_ett.control_dofs_velocity(manipulator_vel, manipulator_jnts_idx)
        arter_ett.control_dofs_velocity(wheel_vel, wheel_jnts_idx)
        arter_ett.control_dofs_velocity(steer_vel, steerring_jnts_idx)

        scene.step()
        if RECORD:
            cam.render(depth=False)
       

    if RECORD:
        cam.stop_recording(save_to_filename="arter_drop.mp4", fps=60)
