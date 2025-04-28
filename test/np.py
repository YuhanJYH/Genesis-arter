import numpy as np
import torch


wheel_vel_mask = np.array([1, 1, -1, -1])
wheel_vel = [3 * np.pi] * 4
wheel_vel = np.array(wheel_vel) * np.array(wheel_vel_mask)
wheel_vel = np.tile(wheel_vel, (1, 1))
print("wheel_vel: ", wheel_vel)
