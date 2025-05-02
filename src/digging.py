import genesis as gs

# gs.init()
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
import math
import pinocchio as pin
import rsl_rl

# local imports
from assets import friction_coefficients
import helpers
import arter

RECORD = False
DEBUG_MODE = False

N_ENVS = 1
ENV_SPACING = 30.0

INIT_POS = (0.0, 0.0, 6.0)
INIT_QUAT = (0.0, 0.0, 0.0, 1.0)

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

################## mesh generating ##################
import numpy as np
import trimesh

# --- Configuration ---
top_width = 10.0  # Width/Length of the top square platform (meters)
height = 5.0  # Height of the platform (meters)
slope_angle_deg = 80.0  # Angle of the slopes in degrees

# --- Calculations ---
# Calculate the horizontal extent of the slope
# For a 45-degree slope, horizontal extent = vertical height
if np.isclose(slope_angle_deg, 45.0):
    horizontal_extent = height
else:
    # General case (though the request specified 45)
    slope_angle_rad = np.radians(slope_angle_deg)
    # Ensure tan(slope_angle_rad) is not zero to avoid division by zero
    if np.isclose(np.tan(slope_angle_rad), 0.0):
        raise ValueError("Slope angle cannot be 0 or 180 degrees.")
    horizontal_extent = height / np.tan(slope_angle_rad)

# Calculate base dimensions
base_width = top_width + 2 * horizontal_extent

# Calculate half-widths for easier coordinate definition (assuming centered at origin)
half_top = top_width / 2.0
half_base = base_width / 2.0

# --- Define Vertices ---
# 8 vertices: 4 for the base, 4 for the top platform
# Vertices are defined as [x, y, z]
vertices = np.array(
    [
        # Base vertices (z=0) - Indices 0 to 3
        [-half_base, -half_base, 0.0],  # 0: Front-Left-Bottom
        [half_base, -half_base, 0.0],  # 1: Front-Right-Bottom
        [half_base, half_base, 0.0],  # 2: Back-Right-Bottom
        [-half_base, half_base, 0.0],  # 3: Back-Left-Bottom
        # Top vertices (z=height) - Indices 4 to 7
        [-half_top, -half_top, height],  # 4: Front-Left-Top
        [half_top, -half_top, height],  # 5: Front-Right-Top
        [half_top, half_top, height],  # 6: Back-Right-Top
        [-half_top, half_top, height],  # 7: Back-Left-Top
    ]
)

# --- Define Faces ---
# Define the faces as triangles using vertex indices.
# 6 faces (top, bottom, 4 sides), each split into 2 triangles = 12 triangles total.
# Winding order (counter-clockwise when viewed from outside) matters for normals.
faces = np.array(
    [
        # Bottom face (z=0) - Triangles using vertices 0, 1, 2, 3
        # Normal should point down (-Z)
        [0, 1, 2],  # Triangle 1
        [0, 2, 3],  # Triangle 2
        # Top face (z=height) - Triangles using vertices 4, 5, 6, 7
        # *** CORRECTED WINDING ORDER *** Normal should point up (+Z)
        [4, 5, 6],  # Triangle 1 (Previously [4, 6, 5])
        [4, 6, 7],  # Triangle 2 (Previously [4, 7, 6])
        # Side faces - Each side is a quad split into two triangles
        # Normals should point outwards from the center.
        # Front face (-Y direction) - Vertices 0, 1, 5, 4
        [0, 1, 5],
        [0, 5, 4],
        # Right face (+X direction) - Vertices 1, 2, 6, 5
        [1, 2, 6],
        [1, 6, 5],
        # Back face (+Y direction) - Vertices 2, 3, 7, 6
        [2, 3, 7],
        [2, 7, 6],
        # Left face (-X direction) - Vertices 3, 0, 4, 7
        [3, 0, 4],
        [3, 4, 7],
    ]
)

platform_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
platform_mesh.fix_normals()
platform_mesh.remove_duplicate_faces()
platform_mesh.remove_unreferenced_vertices()
platform_mesh.remove_infinite_values()
platform_mesh.export(file_type="stl", file_obj="/workspace/genesis/assets/meshes/platform1.stl")

platform_morph = gs.morphs.Mesh(
    file="/workspace/genesis/assets/meshes/platform1.stl",
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
swiviel_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.swiviel_jnts]
stabilizer_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.stabilizer_jnts]
steerring_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.steerring_jnts]
wheel_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.wheel_jnts]
wheel_vel_mask = np.array([1, -1, 1, -1])
wheel_link_names = [
    "arter/wheel_fl_link",
    "arter/wheel_fr_link",
    "arter/wheel_rl_link",
    "arter/wheel_rr_link",
]
claw_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.claw_jnts]
chasis_jnts_idx = swiviel_jnts_idx + stabilizer_jnts_idx + claw_jnts_idx
manipulator_jnts_idx = [arter_ett.get_joint(name).dof_idx_local for name in arter.manipulator_jnts]
whole_jnts_idx = manipulator_jnts_idx + chasis_jnts_idx + wheel_jnts_idx + steerring_jnts_idx

print("friction wheel:", arter_ett.get_link("arter/wheel_fl_link").geoms[0].friction)

# Set the friction coefficient for the wheel links
for wheel_link_name in wheel_link_names:
    wheel_link = arter_ett.get_link(wheel_link_name)
    # wheel_link.set_friction(friction_coefficients["road"]["dry_asphalt_concrete"]["max"])  # Set to min value
    wheel_link.set_friction(10.0)  # Set to min value

print("friction wheel:", arter_ett.get_link("arter/wheel_fl_link").geoms[0].friction)

########################## build ##########################

scene.build(n_envs=N_ENVS, env_spacing=(ENV_SPACING, ENV_SPACING))

if DEBUG_MODE:
    import IPython
    IPython.embed() # start the interactive shell

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

    for i in range(0, iter_num):
        chasis_vel = np.zeros(len(chasis_jnts_idx))
        chasis_vel = np.tile(chasis_vel, (N_ENVS, 1))
        manipulator_vel = np.zeros(len(manipulator_jnts_idx))
        manipulator_vel = np.tile(manipulator_vel, (N_ENVS, 1))

        wheel_vel = np.array([1 * np.pi] * 4)
        wheel_vel = wheel_vel_mask * wheel_vel
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

        base_pos = arter_ett.get_pos()
        base_quat = arter_ett.get_quat()

        # reset to init at tipping
        for i in range(N_ENVS):
            # if arter_entity.get_pos()[i][2] < 5.2 or arter_entity.get_pos()[i][2] > 6.5:
            env_quat = base_quat[i]
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
            if diff_angle > 40 or base_pos[i][2] < 5.2 or base_pos[i][2] > 6.5:
                print("arter is tipping, reset to init")
                # reset to init
                for j in range(len(base_pos[i])):
                    base_pos[i][j] = INIT_POS[j]
                arter_ett.set_pos(base_pos)
                for j in range(len(base_quat[i])):
                    base_quat[i][j] = INIT_QUAT[j]
                arter_ett.set_quat(base_quat)

                # give a new manipulatopr position at new init
                env_manipulator_init_pos = [0, 1, 0.5, 0, 0, 0, 0]
                env_manipulator_init_pos = np.array(env_manipulator_init_pos)
                rand_mask = np.random.uniform(low=-0.5, high=0.5, size=(1, 7))
                manipulator_init_pos[i] = env_manipulator_init_pos + rand_mask
                arter_ett.set_dofs_position(manipulator_init_pos, manipulator_jnts_idx)
                print("randomized manipulator init pos: ", manipulator_init_pos)

    if RECORD:
        cam.stop_recording(save_to_filename="arter_drop.mp4", fps=60)
