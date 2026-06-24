# Teoria — AR Markerless: dalla Geometria al Codice

Documento di studio personale. Copre le derivazioni matematiche dietro ogni stadio della pipeline `ar-markerless`: modello camera, FAST/ORB, homography, RANSAC, PnP.

---

## 1. Il Modello Camera Pinhole

Tutto parte da una domanda: **come si relaziona un punto nel mondo 3D con la sua proiezione 2D sull'immagine?**

### 1.1 Setup geometrico

Immagina un foro stenopeico (pinhole): la luce da un punto 3D $P = (X, Y, Z)$ nel mondo passa attraverso un singolo punto (il centro ottico, origine del sistema camera) e colpisce un piano immagine a distanza $f$ (la lunghezza focale).

Per similitudine di triangoli — il triangolo formato da $P$, il centro ottico, e l'asse $Z$, è simile al triangolo formato dalla sua proiezione $p$, il centro ottico, e il piano immagine:

$$
\frac{x}{f} = \frac{X}{Z} \quad \Rightarrow \quad x = f \frac{X}{Z}
$$

$$
\frac{y}{f} = \frac{Y}{Z} \quad \Rightarrow \quad y = f \frac{Y}{Z}
$$

Questa è la **proiezione prospettica**: dividere per $Z$ è il motivo per cui oggetti più lontani appaiono più piccoli (più $Z$ è grande, più $x$ e $y$ si riducono).

### 1.2 Da coordinate fisiche a pixel

Il piano immagine è misurato in unità fisiche (mm), ma un'immagine digitale è fatta di pixel. Serve convertire: $f_x, f_y$ sono la focale espressa in **pixel** (tiene conto sia della focale fisica che della densità di pixel del sensore), e $(c_x, c_y)$ è il **centro ottico** in pixel (di solito vicino al centro dell'immagine, ma non sempre esattamente).

$$
u = f_x \frac{X}{Z} + c_x, \qquad v = f_y \frac{Y}{Z} + c_y
$$

### 1.3 Forma matriciale: la matrice intrinseca K

Usando **coordinate omogenee** (un trucco che rende le trasformazioni proiettive lineari), scriviamo:

$$
\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix}
=
\underbrace{\begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}}_{K}
\begin{pmatrix} X \\ Y \\ Z \end{pmatrix}
$$

dove poi $u = u'/w'$, $v = v'/w'$ (la divisione finale recupera la divisione per $Z$ della prospettiva).

**Collegamento al codice:** questa è esattamente la matrice `K` che hai costruito nello script, con la stima $f_x = f_y \approx \text{larghezza frame}$. È una stima perché la vera calibrazione richiederebbe foto di una scacchiera nota da più angolazioni (algoritmo di Zhang) per risolvere $K$ esattamente — qui si usa un'approssimazione plausibile per webcam consumer.

### 1.4 Estrinseci: dove sta la camera nel mondo

$K$ descrive *come* la camera proietta, ma non *dove* la camera si trova rispetto al mondo. Questo è descritto da una rotazione $R$ (matrice 3×3) e una traslazione $t$ (vettore 3×1), che insieme trasformano un punto dal sistema di coordinate del mondo a quello della camera:

$$
P_{\text{camera}} = R \cdot P_{\text{mondo}} + t
$$

La proiezione completa, mondo → pixel, è quindi:

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K [R \mid t] \begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix}
$$

dove $\lambda$ è un fattore di scala (la $Z$ della camera, recuperata dividendo).

**Questo è esattamente quello che `solvePnP` calcola**: dati punti 3D noti e le loro proiezioni 2D osservate, trova $R$ e $t$ (cioè dove sta la camera).

---

## 2. FAST Corner Detection (il detector dentro ORB)

ORB usa **FAST** (Features from Accelerated Segment Test) per trovare i keypoint. L'idea: un angolo è un punto dove l'intensità cambia bruscamente in più direzioni.

### 2.1 Il test del cerchio

Per ogni pixel candidato $p$, si considera un cerchio di 16 pixel a raggio fissato (tipicamente 3) attorno a $p$. Sia $I(p)$ l'intensità del pixel centrale, $I(x)$ l'intensità di un pixel sul cerchio, e $t$ una soglia.

Un pixel $x$ sul cerchio è classificato come:

$$
\text{più chiaro se } I(x) > I(p) + t, \qquad \text{più scuro se } I(x) < I(p) - t
$$

**Criterio FAST:** $p$ è un angolo se esistono **almeno 12 pixel contigui** (su 16) tutti più chiari, oppure tutti più scuri, del centro.

### 2.2 Perché funziona geometricamente

Pensa a un vero angolo nell'immagine (es. il vertice di un quadrato chiaro su sfondo scuro). Se il cerchio di 16 pixel attraversa quel vertice, una porzione contigua del cerchio cadrà "dentro" la regione chiara e il resto "fuori" — esattamente il pattern che il test cerca. Su una regione piatta (senza angoli), invece, i pixel del cerchio hanno intensità simile al centro ovunque: nessun arco contiguo soddisfa la soglia.

### 2.3 Accelerazione: il test ad alta velocità

Controllare tutti i 16 pixel per ogni candidato è costoso. FAST usa una scorciatoia: controlla prima solo i pixel a 0°, 90°, 180°, 270° (i 4 punti cardinali del cerchio). Se almeno 3 di questi 4 non soddisfano già la condizione di base, il pixel viene scartato immediatamente, senza testare gli altri 12 — la maggior parte dei pixel "non-angolo" viene scartata in pochi confronti.

**Perché ORB sceglie FAST:** è uno dei detector più veloci esistenti, requisito chiave per il tracking in tempo reale che serve in AR.

---

## 3. Da FAST a "Oriented" FAST: l'invarianza alla rotazione

FAST di per sé non è invariante alla rotazione: se l'immagine ruota, il pattern dei 16 pixel attorno a un angolo cambia ordine. ORB aggiunge un **orientamento** ad ogni keypoint, calcolato tramite il **momento di intensità**.

### 3.1 Momento di intensità e centroide

Per una piccola patch attorno al keypoint, si definisce il momento:

$$
m_{pq} = \sum_{x,y} x^p y^q I(x,y)
$$

Il **centroide** della patch (il "baricentro pesato per intensità") è:

$$
C = \left( \frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}} \right)
$$

### 3.2 L'orientamento come vettore dal centro al centroide

Il vettore dal centro geometrico della patch $O$ al centroide $C$ definisce una direzione dominante:

$$
\theta = \arctan2(m_{01}, m_{10})
$$

**Intuizione geometrica:** se la patch ha più "massa" di intensità da un lato (es. un bordo chiaro-scuro inclinato), il centroide si sposta in quella direzione, e $\theta$ cattura l'orientamento del bordo/struttura locale.

Una volta noto $\theta$, il descriptor (vedi sezione 4) viene calcolato **ruotando virtualmente la patch di $-\theta$** prima del confronto — questo è ciò che rende ORB rotation-invariant: confrontando sempre la patch nel suo "orientamento canonico", la rotazione dell'immagine originale non altera il descriptor.

---

## 4. BRIEF e il Descriptor Binario di ORB

### 4.1 L'idea di BRIEF: confronti binari

Invece di descrivere un punto con valori di intensità continui (come SIFT), BRIEF costruisce un descriptor **binario** facendo $n$ confronti tra coppie di pixel pre-definite $(p_i, q_i)$ in una patch attorno al keypoint:

$$
\tau(p_i, q_i) = \begin{cases} 1 & \text{se } I(p_i) < I(q_i) \\ 0 & \text{altrimenti} \end{cases}
$$

Il descriptor è la concatenazione di questi bit, tipicamente $n = 256$:

$$
f(p) = \sum_{i=1}^{256} 2^{i-1} \, \tau(p_i, q_i)
$$

Risultato: una stringa di 256 bit (32 byte) per ogni keypoint.

### 4.2 Perché funziona

Anche se ogni singolo confronto è "debole" (porta poca informazione), la combinazione di centinaia di confronti su pattern geometrici ben scelti è sorprendentemente discriminativa — due punti diversi nell'immagine produrranno, con alta probabilità, pattern di bit diversi.

### 4.3 ORB = Oriented FAST + Rotated BRIEF

ORB applica BRIEF non sulla patch originale, ma sulla patch **ruotata secondo l'orientamento $\theta$** calcolato nella sezione 3. In pratica, le coordinate delle coppie $(p_i, q_i)$ vengono trasformate con la matrice di rotazione:

$$
\begin{pmatrix} p_i' \\ q_i' \end{pmatrix} = R_\theta \begin{pmatrix} p_i \\ q_i \end{pmatrix}, \qquad R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

Questo è il passo che rende l'intero descriptor robusto a rotazioni dell'immagine (parzialmente, vedi sezione 9 sui limiti).

---

## 5. Distanza di Hamming: perché è la metrica corretta

Per due descriptor binari $A$ e $B$ di $n$ bit, la **distanza di Hamming** conta semplicemente in quante posizioni i bit differiscono:

$$
d_H(A, B) = \sum_{i=1}^{n} (A_i \oplus B_i)
$$

dove $\oplus$ è lo XOR bit a bit (1 se i bit sono diversi, 0 se uguali).

**Perché non la distanza euclidea:** la distanza euclidea ha senso per vettori in uno spazio continuo dove "vicino" significa numericamente simile (caso di SIFT, descriptor float). Per stringhe binarie generate da confronti booleani indipendenti, il concetto naturale di "differenza" è *quanti confronti sono cambiati*, non una distanza geometrica in uno spazio vettoriale continuo. Inoltre, la distanza di Hamming si calcola in hardware con un'istruzione XOR + popcount, **enormemente più veloce** della distanza euclidea — altro motivo per cui ORB+Hamming è la combinazione scelta per il real-time.

---

## 6. Homography: derivazione

### 6.1 Cos'è geometricamente

Una homography $H$ è una matrice 3×3 che mappa punti di un piano nello spazio 2D di un altro piano (qui: il piano del target fisico → il piano immagine del frame), tramite una **trasformazione proiettiva**:

$$
\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}, \qquad H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix}
$$

con le coordinate finali recuperate da $x'' = x'/w'$, $y'' = y'/w'$.

### 6.2 Perché 8 parametri liberi (non 9)

$H$ ha 9 elementi ma è definita **a meno di un fattore di scala** (moltiplicare tutta $H$ per una costante non cambia la trasformazione, perché poi si divide per $w'$). Quindi $H$ ha effettivamente **8 gradi di libertà**.

### 6.3 DLT — Direct Linear Transform: come si stima H da punti corrispondenti

Per ogni coppia di punti corrispondenti $(x_i, y_i) \leftrightarrow (x_i', y_i')$, l'equazione $\begin{pmatrix} x_i' \\ y_i' \\ 1 \end{pmatrix} \times H \begin{pmatrix} x_i \\ y_i \\ 1 \end{pmatrix} = 0$ (prodotto vettoriale nullo, perché i due vettori sono paralleli a meno di scala) si espande in due equazioni lineari nei coefficienti di $H$:

$$
x_i' (h_{31} x_i + h_{32} y_i + h_{33}) - (h_{11} x_i + h_{12} y_i + h_{13}) = 0
$$

$$
y_i' (h_{31} x_i + h_{32} y_i + h_{33}) - (h_{21} x_i + h_{22} y_i + h_{23}) = 0
$$

Ogni coppia di punti corrispondenti fornisce **2 equazioni lineari**. Con 8 incognite (a meno di scala), **servono almeno 4 coppie di punti** (4 × 2 = 8 equazioni) per risolvere il sistema esattamente.

### 6.4 Sovra-determinazione e soluzione ai minimi quadrati

In pratica abbiamo *molti più* di 4 match (decine, dato dal matching ORB). Il sistema diventa sovra-determinato: $A \mathbf{h} = 0$, dove $A$ è una matrice $2n \times 9$ (per $n$ punti) e $\mathbf{h}$ è $H$ "appiattita" in un vettore di 9 elementi.

La soluzione che minimizza l'errore (nel senso dei minimi quadrati, con il vincolo $\|\mathbf{h}\| = 1$ per evitare la soluzione banale $\mathbf{h} = 0$) è data dal **vettore singolare destro corrispondente al valore singolare più piccolo** della SVD (Singular Value Decomposition) di $A$. Questo è ciò che `cv2.findHomography` calcola internamente prima di applicare RANSAC.

---

## 7. RANSAC: derivazione probabilistica

### 7.1 Il problema

Tra i match ORB↔frame, una frazione sono **outlier** (falsi match, rumore). Stimare $H$ usando *tutti* i punti (anche gli outlier) con i minimi quadrati produce una stima distorta — anche pochi outlier estremi possono "tirare" la soluzione lontano dalla verità.

### 7.2 L'idea di RANSAC

Invece di usare tutti i punti insieme, RANSAC:
1. Sceglie un sottoinsieme **minimo casuale** di punti (4, il minimo per la homography)
2. Calcola $H$ da quei 4 punti
3. Conta quanti **altri** punti sono "consistenti" con questa $H$ (errore di riproiezione sotto una soglia — il `5.0` px che hai usato)
4. Ripete molte volte, tenendo la $H$ che ha il maggior numero di consistenti (gli **inliers**)

### 7.3 Quante iterazioni servono? Derivazione

Sia $w$ la probabilità che un punto scelto a caso sia un inlier (cioè $1-w$ è la frazione di outlier). La probabilità che **tutti i 4 punti** di un campionamento siano inlier è:

$$
p_{\text{sample valido}} = w^4
$$

La probabilità che un singolo tentativo **fallisca** (almeno un outlier nel campione) è $1 - w^4$.

Vogliamo che, dopo $k$ tentativi indipendenti, la probabilità di **non aver mai pescato** un campione tutto-inlier sia sotto una soglia accettabile $1-P$ (es. $P = 0.99$, vogliamo il 99% di sicurezza di successo):

$$
(1 - w^4)^k \leq 1 - P
$$

Risolvendo per $k$ (logaritmo da entrambi i lati):

$$
k \geq \frac{\log(1-P)}{\log(1 - w^4)}
$$

**Esempio numerico:** se $w = 0.5$ (metà dei match sono outlier) e vogliamo $P = 0.99$:

$$
k \geq \frac{\log(0.01)}{\log(1 - 0.5^4)} = \frac{\log(0.01)}{\log(0.9375)} \approx \frac{-4.605}{-0.0645} \approx 71
$$

Quindi con metà dei punti outlier, **circa 71 iterazioni** bastano per il 99% di confidenza — questo è il motivo per cui RANSAC è computazionalmente leggero anche con molto rumore: il numero di iterazioni cresce solo logaritmicamente al peggiorare di $w$, non esplosivamente.

### 7.4 Collegamento al codice

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

Il `5.0` è la soglia di errore di riproiezione (in pixel) usata nel passo 3 sopra per decidere se un punto è inlier. `mask` è il vettore binario che segna quali punti sono stati classificati inlier dalla $H$ vincente — è esattamente da qui che viene il numero "inliers" che hai visto in console.

---

## 8. Il Problema PnP (Perspective-n-Point) e solvePnP

### 8.1 Perché homography non basta per il 3D

La homography ti dice come un **piano si trasforma in un altro piano** — è intrinsecamente 2D↔2D. Per sapere **dove sta la camera nello spazio 3D** (rotazione $R$ e traslazione $t$ reali, non solo una trasformazione di piano), serve risolvere il problema PnP.

### 8.2 Formulazione del problema

Dati $n$ punti 3D noti $P_i = (X_i, Y_i, Z_i)$ (nel nostro caso, i 4 angoli del target, con $Z=0$ perché il target è piatto) e le loro proiezioni 2D osservate $p_i = (u_i, v_i)$ nel frame, trovare $R, t$ tali che:

$$
\lambda_i \begin{pmatrix} u_i \\ v_i \\ 1 \end{pmatrix} = K (R \, P_i + t) \quad \text{per ogni } i
$$

Questo è un sistema non lineare (per via della divisione implicita in $\lambda_i$ e della struttura di $R$, che ha vincoli di ortogonalità).

### 8.3 Perché 4 punti complanari bastano (il nostro caso)

Il caso generale di PnP richiede minimo 3 punti (P3P) per una soluzione, ma con ambiguità multiple da risolvere con un quarto punto. **Quando i punti sono complanari** (come i 4 angoli di un target piatto, il nostro caso esatto), il problema si semplifica: esiste una relazione diretta tra la homography del piano e la posa $(R, t)$, perché — per un piano con $Z=0$ — la proiezione si riduce a:

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \begin{pmatrix} r_1 & r_2 & t \end{pmatrix} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}
$$

dove $r_1, r_2$ sono le prime due colonne di $R$ (la terza colonna, relativa a $Z$, si "perde" perché $Z=0$ sempre). Si nota che $K(r_1 \; r_2 \; t)$ **è strutturalmente una homography** — questo è il ponte matematico diretto tra ciò che hai calcolato nello step 3 (homography) e ciò che `solvePnP` formalizza nello step 4 con il vincolo di ortonormalità di $R$ esplicito (che homography da sola non impone).

### 8.4 Come OpenCV risolve (in breve)

`cv2.solvePnP` con il metodo iterativo di default minimizza l'**errore di riproiezione**:

$$
\min_{R, t} \sum_{i=1}^{n} \left\| p_i - \hat{p}_i(R, t) \right\|^2
$$

dove $\hat{p}_i(R,t)$ è la proiezione prevista del punto 3D $i$ usando la posa corrente. Si parte da una stima iniziale (spesso ottenuta proprio dalla relazione homography→posa della sezione 8.3) e si raffina iterativamente (algoritmo di Levenberg-Marquardt), aggiustando $R, t$ per minimizzare l'errore tra dove i punti *dovrebbero* proiettarsi e dove sono *osservati*.

---

## 9. Rodrigues: da matrice di rotazione a vettore compatto

### 9.1 Perché serve

Una matrice di rotazione $R$ ha 9 elementi ma solo 3 gradi di libertà (vincoli di ortogonalità: $R^T R = I$, $\det(R) = 1$). `solvePnP` restituisce `rvec`, un vettore a 3 componenti — rappresentazione compatta che evita di portarsi dietro 9 numeri ridondanti.

### 9.2 La formula di Rodrigues

Un vettore di rotazione $\mathbf{r} = \theta \hat{n}$ codifica **sia l'asse di rotazione** $\hat{n}$ (versore) **sia l'angolo** $\theta$ (la sua lunghezza $\|\mathbf{r}\| = \theta$). La matrice di rotazione corrispondente si ricostruisce con:

$$
R = I + \sin\theta \, [\hat{n}]_\times + (1 - \cos\theta) \, [\hat{n}]_\times^2
$$

dove $[\hat{n}]_\times$ è la **matrice antisimmetrica** associata al prodotto vettoriale con $\hat{n} = (n_x, n_y, n_z)$:

$$
[\hat{n}]_\times = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}
$$

**Intuizione:** questa formula è la versione "chiusa" (senza serie infinita) dell'esponenziale di matrice $R = e^{\theta [\hat{n}]_\times}$, che descrive una rotazione continua attorno all'asse $\hat{n}$.

### 9.3 Perché l'EMA lineare su rvec non è rigoroso (collegamento alla sezione smoothing)

Il vettore di Rodrigues vive in uno spazio dove la somma/media diretta **non corrisponde a una composizione geometrica corretta di rotazioni**, specialmente quando l'angolo $\theta$ è vicino a $\pi$ o quando si passa per discontinuità (es. l'asse $\hat{n}$ può "saltare" di segno per rotazioni equivalenti). Una media pesata di due `rvec` molto diversi non produce necessariamente la rotazione "intermedia" geometricamente sensata.

La soluzione matematicamente corretta è convertire in **quaternioni** $q = (\cos(\theta/2), \sin(\theta/2)\hat{n})$ e usare **SLERP** (Spherical Linear intERPolation), che interpola lungo il cammino più corto sulla sfera unitaria a 4 dimensioni dei quaternioni — garantendo che ogni rotazione intermedia sia una rotazione valida e che il percorso sia il più breve geometricamente. Per piccole variazioni frame-to-frame (il caso pratico del progetto, target quasi fermo), l'approssimazione lineare resta numericamente vicina al risultato corretto — motivo per cui "funziona bene in pratica" nonostante non sia rigorosa.

---

## 10. Mappa Concettuale: dal Pixel alla Posa

```
Frame webcam (pixel)
       │
       ▼
FAST corner detection ──► keypoints (dove sono le feature)
       │
       ▼
Momento di intensità ──► orientamento θ per ogni keypoint
       │
       ▼
BRIEF ruotato di θ ──► descriptor binario a 256 bit (ORB completo)
       │
       ▼
BFMatcher (distanza Hamming) ──► coppie target↔frame candidate
       │
       ▼
RANSAC + DLT ──► Homography H (scarta outlier, stima trasformazione di piano)
       │
       ▼
Homography → stima iniziale posa ──► solvePnP (raffina R, t minimizzando errore riproiezione)
       │
       ▼
Rodrigues: R ↔ rvec (rappresentazione compatta)
       │
       ▼
K [R|t] ──► projectPoints: vertici 3D del modello → pixel sullo schermo
```

---

*Documento di studio personale — non destinato a pubblicazione esterna.*
