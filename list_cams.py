import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Indice {i}: OK, frame {frame.shape}")
        else:
            print(f"Indice {i}: si apre ma non legge frame")
        cap.release()
    else:
        print(f"Indice {i}: non disponibile")