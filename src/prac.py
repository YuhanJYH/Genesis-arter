import genesis as gs
gs.init(backend=gs.gpu)

scene = gs.Scene(
  show_viewer=True,
  viewer_options=gs.options.ViewerOptions(
    res=(640, 480),
    max_FPS=30,
    camera_fov = 40,
  ),
  vis_options=gs.options.VisOptions(
    show_world_frame=True,
    ambient_light=(0.5, 0.5, 0.5),
 ),
  renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
  morph=gs.morphs.Plane()
)

cam = scene.add_camera(
  res= (640, 480),
  pos=(0, 10, 10),
  lookat=(0, 0, 0),
  fov=40,
  GUI=False,
)

# man = scene.add_entity(
#   gs.morphs.MJCF(file = "/workspace/genesis/assets/xml/mujoco/model/hammock/humanoid_body.xml",
#                  pos = (0, 0, 1),),
#   visualize_contact=True,
# )

arter = scene.add_entity(
  gs.morphs.URDF(file = "/workspace/genesis/assets/urdf/arter/arter.urdf",
                 pos = (0, 0, 2),
                 ),
  visualize_contact=True,
)

cylinder = scene.add_entity(
  gs.morphs.Cylinder(radius=0.2, height=2, pos=(0, 2, 1)),
)

scene.build()

# cam.start_recording()

for i in range(1000):
  scene.step()
  # cam.render(depth=True)
  # rgb, depth, seg, normal = cam.render(depth=True, segmentation=False, normal=False)

# cam.stop_recording(save_to_filename="arter_drop.mp4", fps=60)