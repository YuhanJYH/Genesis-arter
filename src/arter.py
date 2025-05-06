import numpy as np

URDF_PATH = "/workspace/genesis/assets/urdf/arter/arter.urdf"
MODEL_PATH = "/workspace/genesis/assets/urdf/arter"

################## Joints ######################
# chasis
stabilizer_jnts = ["stabilizer_fl", "stabilizer_fr", "stabilizer_rl", "stabilizer_rr"]
swiviel_jnts = ["swivel_fl", "swivel_fr", "swivel_rl", "swivel_rr"]
claw_jnts = ["claw_dipper_rl", "claw_dipper_rr", "claw_telescope_rl", "claw_telescope_rr"]
# drive
steerring_jnts = ["steering_fl", "steering_fr", "steering_rl", "steering_rr"]
wheel_jnts = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
# manipulator
manipulator_jnts = ["cabin", "boom", "dipper", "telescope", "shovel", "roto", "tilt"]

################### Links #######################
wheel_link_names = [
    "arter/wheel_fl_link",
    "arter/wheel_fr_link",
    "arter/wheel_rl_link",
    "arter/wheel_rr_link",
]

manipulator_link_names = ["arter/base_link", "arter/cabin_link", "arter/boom_link", "arter/dipper_link", "arter/telescope_link", "arter/shovel_link", "arter/roto_link", "arter/tilt_link","arter/wide_shovel_base_link", "arter/wide_shovel_bottom_link" ]
################### Masks ########################  
wheel_vel_mask = np.array([1, -1, 1, -1])
