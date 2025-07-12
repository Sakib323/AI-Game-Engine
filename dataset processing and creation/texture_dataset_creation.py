# --- This MUST be at the very top before any OpenGL-related import ---
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import trimesh
import pyrender
import cv2

# --- Config ---
GLB_PATH = 'dataset processing and creation/demo_glb.glb'  # path to your GLB file
OUTPUT_DIR = 'albedo_images'
RESOLUTION = 768
CAMERA_DISTANCE = 2.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load mesh ---
mesh = trimesh.load(GLB_PATH, force='scene')
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)

# --- Convert to pyrender mesh and create scene ---
scene = pyrender.Scene()
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
scene.add(pyrender_mesh)

# --- Define camera poses ---
def look_at(eye, target, up):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    z_axis = (target - eye)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = x_axis
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = -z_axis
    matrix[:3, 3] = eye
    return matrix

camera_poses = {
    'front':  look_at([0, 0, CAMERA_DISTANCE], [0, 0, 0], [0, 1, 0]),
    'back':   look_at([0, 0, -CAMERA_DISTANCE], [0, 0, 0], [0, 1, 0]),
    'left':   look_at([-CAMERA_DISTANCE, 0, 0], [0, 0, 0], [0, 1, 0]),
    'right':  look_at([CAMERA_DISTANCE, 0, 0], [0, 0, 0], [0, 1, 0]),
    'top':    look_at([0, CAMERA_DISTANCE, 0], [0, 0, 0], [0, 0, -1]),
    'bottom': look_at([0, -CAMERA_DISTANCE, 0], [0, 0, 0], [0, 0, 1]),
}

# --- Camera and Renderer ---
camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
camera_node = scene.add(camera, pose=np.eye(4))
renderer = pyrender.OffscreenRenderer(RESOLUTION, RESOLUTION)

# --- Render albedo from each view ---
def render_albedo(scene, camera_node, renderer, view_pose):
    scene.set_pose(camera_node, pose=view_pose)
    for light_node in list(scene.light_nodes):
        scene.remove_node(light_node)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = np.copy(view_pose)
    light_pose[:3, 3] += light_pose[:3, 2] * 0.1
    scene.add(light, pose=light_pose)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color[..., :3]

for name, pose in camera_poses.items():
    print(f"Rendering {name} view...")
    albedo = render_albedo(scene, camera_node, renderer, pose)
    out_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    cv2.imwrite(out_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))

renderer.delete()
print(f"✅ Albedo maps saved to: {OUTPUT_DIR}")
