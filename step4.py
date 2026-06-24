import cv2
import numpy as np

# --- Setup target ---
target = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
if target is None:
    raise FileNotFoundError("target.jpg non trovato")

h_target, w_target = target.shape

orb = cv2.ORB_create(nfeatures=1000)
kp_target, desc_target = orb.detectAndCompute(target, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Webcam ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam non accessibile")

ret, first_frame = cap.read()
frame_h, frame_w = first_frame.shape[:2]

# --- Matrice intrinseca stimata (no calibrazione vera) ---
focal_length = frame_w  # stima plausibile per webcam comuni
K = np.array([
    [focal_length, 0, frame_w / 2],
    [0, focal_length, frame_h / 2],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros(4)  # assumiamo nessuna distorsione lente

# --- Punti 3D del target nel suo sistema di coordinate locale (piano z=0) ---
# Il target è un piano: i suoi 4 angoli vivono a z=0, in unità "target" (qui: pixel del target)
# Scaliamo in un'unità arbitraria "world" dividendo per la larghezza, così il cubo ha dimensioni sensate
scale = 1.0 / w_target
object_points_plane = np.float32([
    [0, 0, 0],
    [w_target, 0, 0],
    [w_target, h_target, 0],
    [0, h_target, 0]
]) * scale

# --- Vertici del cubo: poggia sul target (z=0) e si alza (z negativo = verso la camera) ---
cube_size = 0.5  # relativo alla larghezza del target (scala=1)
cz = -cube_size  # asse Z verso la camera è negativo in questa convenzione
cube_points = np.float32([
    [0.25, 0.25, 0], [0.75, 0.25, 0], [0.75, 0.75, 0], [0.25, 0.75, 0],  # base, sul target
    [0.25, 0.25, cz], [0.75, 0.25, cz], [0.75, 0.75, cz], [0.25, 0.75, cz]  # top, sollevato
])

def draw_cube(img, projected_pts):
    pts = np.int32(projected_pts).reshape(-1, 2)
    # base
    img = cv2.drawContours(img, [pts[:4]], -1, (0, 255, 0), 3)
    # lati verticali
    for i in range(4):
        img = cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (0, 255, 0), 3)
    # top
    img = cv2.drawContours(img, [pts[4:]], -1, (0, 255, 0), 3)
    return img

MIN_MATCHES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    status_text = "Target non rilevato"

    if desc_frame is not None and len(kp_frame) > 0:
        matches = bf.match(desc_target, desc_frame)
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < 60]

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

                # Proietta i 4 angoli del target nel frame usando la homography
                # (ci servono come "punti immagine" corrispondenti ai 4 punti 3D noti)
                target_corners_2d = np.float32([
                    [0, 0], [w_target, 0], [w_target, h_target], [0, h_target]
                ]).reshape(-1, 1, 2)
                projected_corners = cv2.perspectiveTransform(target_corners_2d, H)
                image_points = projected_corners.reshape(-1, 2)

                # solvePnP: dai 4 punti 3D noti (piano target) e le loro proiezioni 2D nel frame,
                # ricava rotazione (rvec) e traslazione (tvec) della camera
                success, rvec, tvec = cv2.solvePnP(
                    object_points_plane, image_points, K, dist_coeffs
                )

                if success:
                    # Proietta i vertici del cubo 3D nello spazio immagine usando la posa trovata
                    projected_cube, _ = cv2.projectPoints(
                        cube_points, rvec, tvec, K, dist_coeffs
                    )
                    frame = draw_cube(frame, projected_cube)
                    status_text = f"Cubo agganciato | inliers: {inliers}/{len(good_matches)}"
                else:
                    status_text = "solvePnP fallito"
            else:
                status_text = "Homography fallita"
        else:
            status_text = f"Match insufficienti: {len(good_matches)}/{MIN_MATCHES}"

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Cubo 3D ancorato al target", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()