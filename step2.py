import cv2
import numpy as np

# --- Setup target (una volta sola, fuori dal loop) ---
target = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
if target is None:
    raise FileNotFoundError("target.jpg non trovato")

orb = cv2.ORB_create(nfeatures=1000)
kp_target, desc_target = orb.detectAndCompute(target, None)

# Matcher: ORB usa descriptor binari -> distanza di Hamming
# crossCheck=True: accetta un match solo se è reciproco (A->B e B->A coincidono)
# riduce drasticamente i match falsi positivi
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Webcam ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam non accessibile")

# Leggiamo un frame per conoscere le dimensioni della webcam
ret, sample_frame = cap.read()
frame_h = sample_frame.shape[0]

# Ridimensioniamo il target così la sua altezza è comparabile a quella del frame webcam,
# mantenendo le proporzioni originali -> drawMatches affianca due immagini di peso visivo simile
target_scale = frame_h / target.shape[0]
target_resized = cv2.resize(target, (int(target.shape[1] * target_scale), frame_h))
# I keypoint vanno scalati di conseguenza, altrimenti i pallini non coincidono più con l'immagine
kp_target_resized = [
    cv2.KeyPoint(kp.pt[0] * target_scale, kp.pt[1] * target_scale, kp.size * target_scale)
    for kp in kp_target
]

max_display_width = 1400  # larghezza massima finale a schermo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    if desc_frame is not None and len(kp_frame) > 0:
        matches = bf.match(desc_target, desc_frame)
        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = [m for m in matches if m.distance < 60]

        vis = cv2.drawMatches(target_resized, kp_target_resized, gray_frame, kp_frame,
                               good_matches[:50], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.putText(vis, f"Match buoni: {len(good_matches)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        vis = frame

    # Scala l'intera immagine finale per stare comodamente a schermo
    h, w = vis.shape[:2]
    if w > max_display_width:
        scale_factor = max_display_width / w
        vis = cv2.resize(vis, (int(w * scale_factor), int(h * scale_factor)))

    cv2.imshow("Target <-> Webcam matching", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()