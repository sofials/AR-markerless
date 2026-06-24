import cv2
import numpy as np

# Carica il target in scala di grigi (le feature si calcolano sull'intensità, non sul colore)
target = cv2.imread("target.jpg", cv2.IMREAD_GRAYSCALE)
if target is None:
    raise FileNotFoundError("target.jpg non trovato nella cartella")

# ORB: detector + descriptor veloce e gratis (no licenze, gira in real-time)
orb = cv2.ORB_create(nfeatures=1000)

# keypoints = DOVE sono le feature; descriptors = COM'È fatta ognuna
keypoints, descriptors = orb.detectAndCompute(target, None)

print(f"Trovate {len(keypoints)} feature nel target")

# Disegniamole per vedere cosa cattura l'algoritmo
vis = cv2.drawKeypoints(target, keypoints, None,
                        color=(0, 255, 0),
                        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# Ridimensiona l'immagine mantenendo le proporzioni, max 900px sul lato più lungo
max_dim = 900
h, w = vis.shape[:2]
scale_factor = max_dim / max(h, w)
if scale_factor < 1:
    vis = cv2.resize(vis, (int(w * scale_factor), int(h * scale_factor)))

cv2.imshow("Feature ORB sul target", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()