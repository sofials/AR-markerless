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

# I 4 angoli del target, servono per disegnare il contorno proiettato nel frame
target_corners = np.float32([
    [0, 0], [w_target, 0], [w_target, h_target], [0, h_target]
]).reshape(-1, 1, 2)

# --- Webcam ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam non accessibile")

MIN_MATCHES = 15  # sotto questa soglia, non tentiamo nemmeno la homography

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
            # Estrai le coordinate dei punti corrispondenti
            src_pts = np.float32(
                [kp_target[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # RANSAC: trova la homography scartando gli outlier
            # ransacReprojThreshold: tolleranza in pixel per considerare un punto inlier
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                inliers = mask.sum() if mask is not None else 0

                # Proietta i 4 angoli del target nello spazio del frame usando H
                projected_corners = cv2.perspectiveTransform(target_corners, H)

                # Disegna il contorno agganciato al target rilevato
                frame = cv2.polylines(frame, [np.int32(projected_corners)],
                                       True, (0, 255, 0), 3)

                status_text = f"Target rilevato | inliers: {inliers}/{len(good_matches)}"
            else:
                status_text = "Homography fallita"
        else:
            status_text = f"Match insufficienti: {len(good_matches)}/{MIN_MATCHES}"

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Pose estimation - Homography", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()