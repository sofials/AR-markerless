import cv2
import numpy as np

# --- Parser OBJ minimale: solo vertici e facce ---
def load_obj(path):
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(idx)
    return np.float32(vertices), faces

model_vertices, model_faces = load_obj("pyramid.obj")
print(f"Modello caricato: {len(model_vertices)} vertici, {len(model_faces)} facce")

model_scale = 0.4
model_offset = np.float32([0.5, 0.5, 0])
model_points = model_vertices * model_scale + model_offset

# --- CLAHE: equalizzazione adattiva locale, mitiga variazioni di illuminazione ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- Setup target ---
target_raw = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
if target_raw is None:
    raise FileNotFoundError("target.jpg non trovato")

# Applichiamo CLAHE anche al target, una volta, così descriptor target e frame
# sono calcolati nelle stesse condizioni di contrasto normalizzato
target = clahe.apply(target_raw)

h_target, w_target = target.shape

# Più feature: più candidati per RANSAC quando alcuni si perdono per luce/rotazione
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

def draw_model(img, projected_pts, faces):
    pts = np.int32(projected_pts).reshape(-1, 2)
    for face in faces:
        face_pts = pts[face]
        img = cv2.polylines(img, [face_pts], True, (0, 255, 0), 2)
    return img

smoothed_rvec = None
smoothed_tvec = None
alpha = 0.3

# Soglia leggermente più permissiva: con più feature e CLAHE, alcuni match
# "onesti" potrebbero avere distanza un po' più alta del solito 60
MATCH_DISTANCE_THRESHOLD = 70
MIN_MATCHES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = clahe.apply(gray_frame)  # stessa normalizzazione del target

    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    status_text = "Target non rilevato"

    if desc_frame is not None and len(kp_frame) > 0:
        matches = bf.match(desc_target, desc_frame)
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < MATCH_DISTANCE_THRESHOLD]

        if len(good_matches) >= MIN_MATCHES:
            src_pts = np.float32(
                [kp_target[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

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

                    projected_model, _ = cv2.projectPoints(
                        model_points, smoothed_rvec, smoothed_tvec, K, dist_coeffs
                    )
                    frame = draw_model(frame, projected_model, model_faces)
                    status_text = f"Modello agganciato | inliers: {inliers}/{len(good_matches)}"
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

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Modello 3D ancorato (CLAHE + smoothing)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()