# AR Markerless from Scratch

> 🇮🇹 *Per la versione italiana, [scorri qui sotto](#-ar-markerless-da-zero) o clicca nel sommario.*

A from-scratch Python/OpenCV implementation of a markerless AR pipeline — feature detection, matching, homography estimation, 3D pose recovery, and real 3D mesh rendering anchored to a flat image target via webcam.

Built as a bridge project between applied AR/WebAR development (Unity, MindAR, A-Frame) and the Computer Vision fundamentals underneath every markerless AR library.

---

## 📋 Table of Contents

- [English](#ar-markerless-from-scratch)
- [Italiano](#-ar-markerless-da-zero)

---

## What it does

Point a webcam at a flat image target (printed, or shown on a second screen), and the pipeline:

1. Detects distinctive feature points in both the target image and the live frame (**ORB**)
2. Matches them between target and frame (**BFMatcher**, Hamming distance)
3. Estimates how the target plane is transformed in the frame (**Homography + RANSAC**)
4. Recovers the camera's real 3D pose relative to the target (**solvePnP**)
5. Renders a 3D model anchored to that pose — first as a hand-drawn wireframe, then as a fully textured, lit mesh (**pyrender** + **trimesh**)

The result: a 3D object (a rubber duck, in the final demo) that appears to stand on the target and stays correctly anchored as the camera or the target moves.

## Why

Every markerless AR experience — MindAR, ARKit, ARCore, or a proprietary SDK — solves the same underlying problem: recognize a flat target in the real world and anchor virtual content to it, keeping the anchor stable as the camera moves. Having worked with MindAR/WebAR at the application level, the goal here was to invert the perspective: rebuild that CV pipeline from scratch to understand the *how*, not just the *use*.

## Pipeline overview

```
Webcam frame
   │
   ▼
ORB feature detection ──► keypoints + binary descriptors
   │
   ▼
BFMatcher (Hamming distance) ──► candidate target↔frame matches
   │
   ▼
RANSAC + Homography ──► robust plane transformation, outliers discarded
   │
   ▼
solvePnP ──► real 3D camera pose (rotation + translation)
   │
   ▼
EMA smoothing ──► stabilized pose across frames
   │
   ▼
pyrender (OpenGL offscreen) ──► textured, lit 3D mesh composited on the frame
```

## Key technical points

- **ORB over SIFT**: license-free, faster — required for real-time tracking
- **RANSAC**: discards outlier matches when estimating the homography; the inlier/match ratio is a direct quality signal of the tracking at any given frame
- **Homography → solvePnP bridge**: for a planar target, a homography is structurally equivalent to `K[r1 r2 t]` — this is the mathematical link between the 2D plane transform and the actual 3D camera pose
- **Real ↔ virtual camera alignment** (the hardest part of the rendering extension): matching `pyrender`'s virtual camera intrinsics/extrinsics to the real webcam's estimated pose, including an explicit OpenCV↔OpenGL axis-convention conversion and a world→camera to camera→world matrix inversion

## Known limitations (found empirically, not just predicted)

- **Feature distinctiveness vs. quantity**: repetitive patterns (e.g. a checkerboard) produce *more* ORB features but *worse* matching, because the features are ambiguous relative to each other
- **Screen-displayed vs. physical targets**: a target shown on a screen tracks noticeably better than the same content printed — no reflections, no lighting variance, consistent contrast
- **Rotation + lighting gradient**: rotating a physical target combines perspective foreshortening with a directional lighting gradient (one side in shadow, the other overexposed) — CLAHE helps marginally; a from-scratch illumination-normalization attempt actually made tracking *worse* and was reverted
- **No multi-frame tracking**: every frame re-detects from scratch — no optical flow — so tracking quality is fully dependent on that single frame's lighting/match conditions

## Stack

Python 3.11 · OpenCV (`opencv-contrib-python`) · NumPy · `trimesh` · `pyrender`

## Project structure (suggested)

```
ar-markerless/
├── step1.py          # ORB feature detection on the target
├── step2.py          # target ↔ webcam matching visualization
├── step3.py          # homography + RANSAC, 2D contour overlay
├── step4.py          # solvePnP, wireframe cube anchored in 3D
├── step5.py          # OBJ parser, EMA smoothing, CLAHE
├── step6.py          # pyrender integration: textured mesh, real lighting
├── target.jpg        # flat image target
└── *.obj / textures   # 3D model + PBR textures
```

## Next steps

The "re-detect from scratch every frame" limitation is the natural bridge to **optical flow**-based tracking — reserving full re-detection for when tracking is lost. This is the conceptual seed for the next project in the series (mini-SLAM / visual odometry).

---

---

# 🇮🇹 AR Markerless da Zero

Un'implementazione da zero, in Python/OpenCV, di una pipeline AR markerless — feature detection, matching, stima della homography, recupero della posa 3D, e rendering di una mesh 3D vera ancorata a un target piatto via webcam.

Costruito come progetto-ponte tra lo sviluppo AR/WebAR applicato (Unity, MindAR, A-Frame) e i fondamenti di Computer Vision che stanno sotto ogni libreria AR markerless.

## Cosa fa

Punta una webcam su un target piatto (stampato, o mostrato su un secondo schermo), e la pipeline:

1. Rileva punti caratteristici distintivi sia nell'immagine target che nel frame live (**ORB**)
2. Li confronta tra target e frame (**BFMatcher**, distanza di Hamming)
3. Stima come il piano del target è trasformato nel frame (**Homography + RANSAC**)
4. Recupera la posa 3D reale della camera rispetto al target (**solvePnP**)
5. Renderizza un modello 3D ancorato a quella posa — prima come wireframe disegnato a mano, poi come mesh completa, texturizzata e illuminata (**pyrender** + **trimesh**)

Il risultato: un oggetto 3D (una papera di gomma, nella demo finale) che sembra stare in piedi sul target e resta correttamente ancorato mentre la camera o il target si muovono.

## Perché

Ogni esperienza AR markerless — MindAR, ARKit, ARCore, o un SDK proprietario — risolve lo stesso problema di fondo: riconoscere un target piatto nel mondo reale e ancorarci contenuto virtuale, mantenendo l'aggancio stabile mentre la camera si muove. Avendo lavorato con MindAR/WebAR a livello applicativo, l'obiettivo qui era invertire la prospettiva: ricostruire quella pipeline CV da zero per capire il "come", non solo l'"uso".

## Panoramica della pipeline

```
Frame webcam
   │
   ▼
ORB feature detection ──► keypoints + descriptor binari
   │
   ▼
BFMatcher (distanza Hamming) ──► coppie candidate target↔frame
   │
   ▼
RANSAC + Homography ──► trasformazione di piano robusta, outlier scartati
   │
   ▼
solvePnP ──► posa 3D reale della camera (rotazione + traslazione)
   │
   ▼
Smoothing EMA ──► posa stabilizzata tra i frame
   │
   ▼
pyrender (OpenGL offscreen) ──► mesh 3D texturizzata e illuminata, compositata sul frame
```

## Punti tecnici chiave

- **ORB invece di SIFT**: gratuito da licenza, più veloce — requisito per il tracking in tempo reale
- **RANSAC**: scarta i match outlier nella stima della homography; il rapporto inliers/matches è un indicatore diretto della qualità del tracking in un dato istante
- **Ponte Homography → solvePnP**: per un target piatto, una homography è strutturalmente equivalente a `K[r1 r2 t]` — questo è il collegamento matematico tra la trasformazione di piano 2D e la vera posa 3D della camera
- **Allineamento camera reale ↔ virtuale** (la parte più delicata dell'estensione rendering): far coincidere gli intrinseci/estrinseci della camera virtuale di `pyrender` con la posa stimata della webcam reale, incluse una conversione esplicita di convenzione assi OpenCV↔OpenGL e l'inversione della matrice mondo→camera in camera→mondo

## Limiti noti (scoperti empiricamente, non solo previsti)

- **Distintività vs quantità delle feature**: pattern ripetitivi (es. una scacchiera) producono *più* feature ORB ma *peggior* matching, perché le feature sono ambigue tra loro
- **Target su schermo vs fisico**: un target mostrato su schermo traccia notevolmente meglio dello stesso contenuto stampato — nessun riflesso, nessuna variazione di luce, contrasto costante
- **Rotazione + gradiente di illuminazione**: ruotare un target fisico combina il foreshortening prospettico con un gradiente di luce direzionale (un lato in ombra, l'altro sovraesposto) — CLAHE aiuta marginalmente; un tentativo di normalizzazione dell'illuminazione scritto da zero ha effettivamente *peggiorato* il tracking ed è stato annullato
- **Nessun tracking multi-frame**: ogni frame ri-rileva da zero — niente optical flow — quindi la qualità del tracking dipende interamente dalle condizioni di luce/match di quel singolo frame

## Stack

Python 3.11 · OpenCV (`opencv-contrib-python`) · NumPy · `trimesh` · `pyrender`

## Struttura del progetto (suggerita)

```
ar-markerless/
├── step1.py          # feature detection ORB sul target
├── step2.py          # visualizzazione matching target ↔ webcam
├── step3.py          # homography + RANSAC, contorno 2D
├── step4.py          # solvePnP, cubo wireframe ancorato in 3D
├── step5.py          # parser OBJ, smoothing EMA, CLAHE
├── step6.py          # integrazione pyrender: mesh texturizzata, luce reale
├── target.jpg        # immagine target piatta
└── *.obj / textures   # modello 3D + texture PBR
```

## Prossimi passi

Il limite del "ri-rilevare da zero ogni frame" è il ponte naturale verso il tracking basato su **optical flow** — riservando la ri-detection completa solo a quando il tracking si perde. È il seme concettuale del prossimo progetto della serie (mini-SLAM / visual odometry).
