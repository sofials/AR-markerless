import numpy as np
import pyrender
import trimesh

# Una semplice sfera, giusto per vedere se il rendering funziona
sphere = trimesh.creation.icosphere(radius=0.5)
mesh = pyrender.Mesh.from_trimesh(sphere)

scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 0, 1]
])
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=camera_pose)

r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)
r.delete()

import cv2
cv2.imwrite("test_render.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
print("Render completato, salvato in test_render.png")
print("Shape:", color.shape)