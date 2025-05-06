import pinocchio as pin
from pinocchio.utils import *
import numpy as np
import arter

################# set up pinocchio model ##################
model, collision_model, visual_model = pin.buildModelsFromUrdf(arter.URDF_PATH, arter.MODEL_PATH)
data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)


swiviel_jnts_idx = [model.getJointId(name) for name in arter.swiviel_jnts]
stabilizer_jnts_idx = [model.getJointId(name) for name in arter.stabilizer_jnts]
steering_jnts_idx = [model.getJointId(name) for name in arter.steerring_jnts]
wheel_jnts_idx = [model.getJointId(name) for name in arter.wheel_jnts]
claw_jnts_idx = [model.getJointId(name) for name in arter.claw_jnts]
chasis_jnts_idx = swiviel_jnts_idx + stabilizer_jnts_idx + claw_jnts_idx
manipulator_jnts_idx = [model.getJointId(name) for name in arter.manipulator_jnts]

print("manipulator jnts: ", manipulator_jnts_idx)
print("joints: ")
for name in model.names:
    print(f"-{name}")

manipulator_links_idx = [model.getFrameId(name) for name in arter.manipulator_link_names]
print(manipulator_links_idx)
base_link_id = model.getFrameId("arter/base_link")
print(base_link_id)
tip_link_id = model.getFrameId("arter/wide_shovel_bottom_link")
print(tip_link_id)

print("whole body dofs: ", len(model.joints))
print("whole body links: ", len(model.frames))
q = pin.neutral(model)

# Test: set boom joint
# pin.forwardKinematics(model=model, data=data, q=q)
# pin.updateFramePlacements(model=model, data=data)
# pin.updateGeometryPlacements(model, data, collision_model, collision_data)
# # pin.updateGeometryPlacements(model, data, visual_model, visual_data)
# print("tip pos:", data.oMf[tip_link_id])
# print("CoM :", pin.centerOfMass(model, data))

# q[model.idx_qs[manipulator_jnts_idx[2]]] = 0.5
# pin.forwardKinematics(model=model, data=data, q=q)
# pin.updateFramePlacements(model=model, data=data)
# # pin.updateGeometryPlacements(model, data, collision_model, collision_data)
# print("tip pos:", data.oMf[tip_link_id])
# print("CoM :", pin.centerOfMass(model, data))


# q[model.idx_qs[manipulator_jnts_idx[2]]] = 0.8
# pin.forwardKinematics(model=model, data=data, q=q)
# pin.updateFramePlacements(model=model, data=data)
# # pin.updateGeometryPlacements(model, data, collision_model, collision_data)
# print("tip pos:", data.oMf[tip_link_id])
# print("CoM :", pin.centerOfMass(model, data))



# # Print out the placement of each visual geometry object
# print("\nVisual object placements:")
# for k, oMg in enumerate(visual_data.oMg):
#     print("{:d} : {: .2f} {: .2f} {: .2f}".format(k, *oMg.translation.T.flat))

# print("CoM: ", model.com())
# for jnt in model.joints:
#   print("-", jnt)

def compute_mnp_jac(model, data, q, tip_link_name):
    tip_link_id = model.getFrameId(tip_link_name)
    J_full = pin.computeFrameJacobian(model, data, q, tip_link_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_manip = J_full[:,1:8]  
    J_manip[:, [-1, -2]] = J_manip[:, [-2, -1]]  # swap last 2 columns (roto, tilt)
    return J_manip

def set_mnp_jnts_pos(positions, q):
    for idx, pos in enumerate(positions):
        q[model.idx_qs[manipulator_jnts_idx[idx]]] = pos

def set_stabilizer_jnts_pos(positions, q):
    for idx, pos in enumerate(positions):
        q[model.idx_qs[stabilizer_jnts_idx[idx]]] = pos

def set_swivel_jnts_pos(positions, q):
    for idx, pos in enumerate(positions):
        q[model.idx_qs[swiviel_jnts_idx[idx]]] = pos
 
def set_steering_jnts_pos(positions, q):
    for idx, pos in enumerate(positions):
        q[model.idx_qs[steering_jnts_idx[idx]]] = pos

def set_claw_jnts_pos(positions, q):
    for idx, pos in enumerate(positions):
        q[model.idx_qs[claw_jnts_idx[idx]]] = pos
        

