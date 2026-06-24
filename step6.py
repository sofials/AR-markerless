import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image

# --- Setup target ---
target_raw = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
if target_raw is None:
    raise FileNotFoundError("target.jpg non trovato")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
target = clahe.apply(target_raw)
h_target, w_target = target.shape

orb = cv2.ORB_create(nfeatures=2000)
kp_target, desc_target = orb.detectAndCompute(target, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Webcam ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam non accessibile")

ret, first_frame = cap.read()
frame_h, frame_w = first_frame.shape[:2]

focal_length = frame_w
K = np.array([
    [focal_length, 0, frame_w / 2],
    [0, focal_length, frame_h / 2],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros(4)

scale = 1.0 / w_target
object_points_plane = np.float32([
    [0, 0, 0], [w_target, 0, 0], [w_target, h_target, 0], [0, h_target, 0]
]) * scale

target_corners_2d = np.float32([
    [0, 0], [w_target, 0], [w_target, h_target], [0, h_target]
]).reshape(-1, 1, 2)

# --- Caricamento mesh papera ---
duck_trimesh = trimesh.load("RubberDuck_LOD0.obj")

if isinstance(duck_trimesh, trimesh.Scene):
    duck_trimesh = trimesh.util.concatenate(
        [g for g in duck_trimesh.geometry.values()]
    )

# Centra sull'origine
duck_trimesh.apply_translation(-duck_trimesh.centroid)

# Fix orientamento: il modello esportato ha convenzione assi diversa, ruotiamo 180° su X
flip_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
duck_trimesh.apply_transform(flip_x)

# Ruota la papera per farla stare "in piedi" perpendicolare al target, invece che sdraiata
stand_up = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
duck_trimesh.apply_transform(stand_up)

# Scala alla dimensione voluta e posiziona sul target
bounds = duck_trimesh.bounds
size = bounds[1] - bounds[0]
max_extent = max(size)
target_extent = 0.5

duck_trimesh.apply_scale(target_extent / max_extent)
duck_trimesh.apply_translation([0.5, 0.5, -target_extent * 0.5])

# --- Caricamento texture PBR separate ---
def load_texture(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

base_color_img = load_texture("RubberDuck_BaseColor.png")
metallic_img = load_texture("RubberDuck_Metallic.png")
normal_img = load_texture("RubberDuck_Normal.png")
roughness_img = load_texture("RubberDuck_Roughness.png")

material = pyrender.MetallicRoughnessMaterial(
    baseColorTexture=pyrender.Texture(source=base_color_img, source_channels="RGB"),
    metallicRoughnessTexture=pyrender.Texture(source=roughness_img, source_channels="RGB"),
    normalTexture=pyrender.Texture(source=normal_img, source_channels="RGB"),
)

duck_mesh = pyrender.Mesh.from_trimesh(duck_trimesh, material=material)

scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
duck_node = scene.add(duck_mesh)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
light_node = scene.add(light, pose=np.eye(4))

camera = pyrender.IntrinsicsCamera(
    fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
    znear=0.01, zfar=100.0
)
camera_node = scene.add(camera, pose=np.eye(4))

renderer = pyrender.OffscreenRenderer(frame_w, frame_h)

cv_to_gl = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

def solvepnp_pose_to_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvec.flatten()
    cam_to_world_cv = np.linalg.inv(world_to_cam)
    cam_to_world_gl = cam_to_world_cv @ cv_to_gl
    return cam_to_world_gl

MIN_MATCHES = 15
smoothed_rvec = None
smoothed_tvec = None
alpha = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = clahe.apply(gray_frame)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    status_text = "Target non rilevato"
    rendered_frame = frame.copy()

    if desc_frame is not None and len(kp_frame) > 0:
        matches = bf.match(desc_target, desc_frame)
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < 70]

        if len(good_matches) >= MIN_MATCHES:
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                inliers = int(mask.sum()) if mask is not None else 0
                projected_corners = cv2.perspectiveTransform(target_corners_2d, H)
                image_points = projected_corners.reshape(-1, 2)

                success, rvec, tvec = cv2.solvePnP(
                    object_points_plane, image_points, K, dist_coeffs
                )

                if success:
                    if smoothed_rvec is None:
                        smoothed_rvec = rvec
                        smoothed_tvec = tvec
                    else:
                        smoothed_rvec = alpha * rvec + (1 - alpha) * smoothed_rvec
                        smoothed_tvec = alpha * tvec + (1 - alpha) * smoothed_tvec

                    camera_pose_gl = solvepnp_pose_to_camera_pose(smoothed_rvec, smoothed_tvec)
                    scene.set_pose(camera_node, pose=camera_pose_gl)
                    scene.set_pose(light_node, pose=camera_pose_gl)

                    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

                    mask_render = depth > 0
                    color_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
                    rendered_frame = frame.copy()
                    rendered_frame[mask_render] = color_bgr[mask_render]

                    status_text = f"Papera agganciata | inliers: {inliers}/{len(good_matches)}"
                else:
                    status_text = "solvePnP fallito"
                    smoothed_rvec = None
                    smoothed_tvec = None
            else:
                status_text = "Homography fallita"
                smoothed_rvec = None
                smoothed_tvec = None
        else:
            status_text = f"Match insufficienti: {len(good_matches)}/{MIN_MATCHES}"
            smoothed_rvec = None
            smoothed_tvec = None
    else:
        status_text = "Target non rilevato"
        smoothed_rvec = None
        smoothed_tvec = None

    cv2.putText(rendered_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Papera 3D ancorata", rendered_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
renderer.delete()
cv2.destroyAllWindows()