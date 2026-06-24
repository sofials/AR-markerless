# Teoria CV per AR Markerless — Da Zero

Documento di studio personale. Hai già algebra lineare e trigonometria solide — quello che manca è il **vocabolario specifico della Computer Vision**: come si usano quei concetti per risolvere "dove sta la camera" e "cosa vede". Questo documento costruisce ogni passaggio esplicitamente, senza saltare.

---

## 1. Il problema di fondo, in parole semplici

Hai una **foto** (il target) e un **video live** (la webcam). Vuoi sapere: *"il target che conosco, dov'è in questo frame del video, e da che angolazione lo sto guardando?"*

Per rispondere, la pipeline fa 4 cose, in questo ordine logico:

1. **Trova punti riconoscibili** sia nella foto che nel frame (es. un angolo di un oggetto è più riconoscibile di un punto in mezzo a una parete vuota)
2. **Confronta** quei punti per capire quali si corrispondono (questo angolo nella foto = questo angolo nel frame)
3. Dalle corrispondenze, **calcola la trasformazione geometrica** che spiega come il target si è spostato/ruotato/rimpicciolito tra foto e frame
4. Usa quella trasformazione per **disegnare qualcosa** (il cubo, la piramide) nella posizione giusta

Il resto del documento spiega *come* si fa ciascuno di questi 4 passi, dalla geometria di base in su.

---

## 2. Come una camera "vede": il modello pinhole

### 2.1 Partiamo da un'osservazione fisica

Se metti l'occhio vicino a un piccolissimo foro in una scatola buia (camera oscura/pinhole), e dall'altra parte c'è uno schermo, vedrai una proiezione (rovesciata) del mondo esterno. Questo è il modello più semplice di come funziona una camera: la luce da un punto nel mondo passa per **un singolo punto** (il "buco", che chiamiamo centro ottico) e arriva su un piano (il sensore/schermo).

### 2.2 Disegniamolo con coordinate

Metti l'origine del tuo sistema di coordinate **nel centro ottico** della camera. L'asse $Z$ punta nella direzione in cui guarda la camera (in avanti). Un punto nel mondo reale ha coordinate $(X, Y, Z)$ — pensa a $X$ come "quanto a destra", $Y$ come "quanto in alto", $Z$ come "quanto lontano dalla camera".

Il piano immagine sta a una certa distanza $f$ dal centro ottico, perpendicolare all'asse $Z$.

### 2.3 Triangoli simili: la formula della prospettiva

Immagina di guardare la scena di lato (vista dall'alto, ignorando $Y$ per un momento). Hai:
- Un punto nel mondo a distanza $X$ dall'asse centrale, a distanza $Z$ dalla camera
- La sua proiezione sul piano immagine, a distanza $x$ dall'asse centrale, a distanza $f$ dalla camera (perché il piano immagine è a distanza $f$)

Questi due triangoli (quello "grande" nel mondo, quello "piccolo" sull'immagine) sono **simili** — hanno gli stessi angoli, solo scalati. Per triangoli simili, i rapporti dei lati corrispondenti sono uguali:

$$
\frac{x}{f} = \frac{X}{Z}
$$

Risolvendo per $x$ (quello che vogliamo: dove cade il punto sull'immagine):

$$
x = f \cdot \frac{X}{Z}
$$

Stessa cosa per la coordinata verticale:

$$
y = f \cdot \frac{Y}{Z}
$$

**Cosa ci dice questa formula, intuitivamente:** se $Z$ (la distanza) raddoppia, $x$ si dimezza — l'oggetto appare più piccolo se è più lontano. Questo è semplicemente "la prospettiva" formalizzata: è il motivo per cui due rotaie parallele sembrano convergere in lontananza.

### 2.4 Da "unità del mondo" a pixel

Il problema: $x, y$ sopra sono in qualunque unità fisica usi per $X, Y, Z, f$ (es. millimetri). Ma un'immagine digitale è fatta di **pixel**, una griglia discreta. Dobbiamo convertire.

Chiamiamo $f_x$ la lunghezza focale **espressa in pixel orizzontali** (tiene conto sia della focale fisica dell'obiettivo, sia di quanto sono "fitti" i pixel sul sensore — più pixel per millimetro, più $f_x$ è grande per la stessa focale fisica). Stessa cosa per $f_y$ in verticale.

Inoltre, i pixel si contano di solito dall'angolo in alto a sinistra dell'immagine, non dal centro. Quindi serve uno **spostamento** $(c_x, c_y)$ per portare l'origine dal centro ottico (dove la matematica è comoda) all'angolo dell'immagine (dove i pixel sono effettivamente numerati):

$$
u = f_x \cdot \frac{X}{Z} + c_x
$$
$$
v = f_y \cdot \frac{Y}{Z} + c_y
$$

$(u, v)$ sono ora le **coordinate pixel reali** dove il punto $(X,Y,Z)$ del mondo appare nell'immagine. $(c_x, c_y)$ è tipicamente vicino al centro dell'immagine (es. se l'immagine è 1920×1080, $c_x \approx 960$, $c_y \approx 540$).

### 2.5 Perché scrivere tutto come una matrice (e cosa sono le "coordinate omogenee")

Le formule sopra hanno una divisione per $Z$, che è una trasformazione **non lineare** — e le matrici sanno fare solo trasformazioni lineari. Il trucco usato in CV è rappresentare punti con **una coordinata in più** (coordinate omogenee): un punto 2D $(u,v)$ si scrive come $(u, v, 1)$, un punto 3D $(X,Y,Z)$ a volte si estende a $(X,Y,Z,1)$.

Perché aiuta: con questo trucco, la divisione per $Z$ diventa "dividi tutto il vettore risultato per la sua ultima componente", che possiamo fare *dopo* aver applicato una matrice — quindi la moltiplicazione di matrice resta lineare, e la non-linearità (la divisione) si applica come ultimo passo separato.

Definiamo la **matrice intrinseca** $K$:

$$
K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

E scriviamo:

$$
\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix} = K \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}
$$

Poi recuperiamo le coordinate pixel reali dividendo per l'ultima componente: $u = u'/w'$, $v = v'/w'$. Puoi verificare che questo dia esattamente le stesse formule della sezione 2.4 — è solo un modo più compatto e "matriciale" di scrivere la stessa cosa, comodo perché ora possiamo concatenare trasformazioni moltiplicando matrici, invece di riscrivere formule ogni volta.

**Nel tuo codice:** questa è esattamente la matrice `K` che costruisci con `focal_length = frame_width`. È una *stima*, non una misura — la vera calibrazione richiede un procedimento a parte (foto di una scacchiera da più angoli, l'algoritmo si chiama "calibrazione di Zhang"), ma per uno script dimostrativo l'approssimazione va bene.

### 2.6 "Dove sta la camera nello spazio?" — rotazione e traslazione

$K$ ti dice come la camera *proietta* (la sua "lente"). Ma non dice *dove* la camera si trova e *come è orientata* nello spazio rispetto al mondo (o, equivalentemente, rispetto al target).

Per descrivere questo, servono due cose che probabilmente già conosci dall'algebra lineare:
- Una **matrice di rotazione** $R$ (3×3): ruota un vettore senza cambiarne la lunghezza
- Un **vettore di traslazione** $t$ (3×1): sposta un punto

La trasformazione completa da "coordinate nel sistema del mondo" a "coordinate nel sistema della camera" è:

$$
P_{\text{camera}} = R \cdot P_{\text{mondo}} + t
$$

**Intuizione:** se il target è fermo nel mondo e la camera si muove, $R$ e $t$ cambiano frame per frame, descrivendo come la camera si è spostata/ruotata. Questo è **esattamente** quello che `solvePnP` calcola: dati punti del target di cui conosci la posizione 3D, e dove appaiono nel frame (in pixel), trova $R$ e $t$ — cioè "dove e come sta guardando la camera in questo istante".

---

## 3. Cosa rende un punto "riconoscibile"? FAST corner detection

### 3.1 Il problema: quali pixel scegliere?

Un'immagine ha migliaia o milioni di pixel. Non possiamo confrontare *ogni* pixel del target con *ogni* pixel del frame — sarebbe troppo lento, e comunque la maggior parte dei pixel (es. un'area di colore uniforme) non è distintiva: un pixel grigio in mezzo a un muro grigio è identico a mille altri pixel grigi vicini, non c'è modo di "matcharlo" con sicurezza.

Vogliamo trovare pixel **distintivi** — punti dove c'è qualcosa di geometricamente particolare, tipicamente un **angolo**.

### 3.2 Perché un angolo è speciale (e un bordo non basta)

Pensa a un bordo dritto (es. il confine tra una zona chiara e una scura, una linea retta). Se ti muovi *lungo* il bordo, l'aspetto locale non cambia — non sai dire "sono qui" o "sono un centimetro più in là sul bordo", perché localmente è tutto uguale. Il bordo da solo non è sufficiente per localizzare un punto con precisione.

Un **angolo** (dove due bordi si incontrano) è diverso: se ti muovi in *qualsiasi* direzione da un angolo, l'aspetto locale cambia. Questo lo rende un punto che puoi effettivamente localizzare con precisione e ri-riconoscere — è il motivo per cui i detector di feature cercano angoli, non bordi o aree uniformi.

### 3.3 Come FAST testa se un pixel è un angolo

Per ogni pixel candidato $p$ nell'immagine, FAST guarda un **cerchio di 16 pixel** disposti attorno a $p$ a una piccola distanza fissa (immagina un piccolo orologio di 16 "tacche" attorno al pixel centrale).

Sia $I(p)$ il valore di intensità (luminosità, 0=nero, 255=bianco in scala di grigi) del pixel centrale. Per ogni pixel $x$ sul cerchio, lo confrontiamo con $I(p)$ usando una soglia $t$ (es. $t=20$):

- $x$ è "**più chiaro**" se $I(x) > I(p) + t$
- $x$ è "**più scuro**" se $I(x) < I(p) - t$
- altrimenti è "**simile**" (non conta in nessuna delle due categorie)

**Il criterio per dichiarare $p$ un angolo:** deve esistere un **arco contiguo di almeno 12 pixel** (su 16) sul cerchio, tutti classificati "più chiari" (oppure tutti "più scuri") del centro.

### 3.4 Perché questo specifico criterio individua un angolo

Immagina il vertice di un quadrato chiaro su sfondo scuro, e il pixel candidato $p$ è esattamente su quel vertice. Il cerchio di 16 pixel attorno a $p$ attraverserà sia la regione chiara (dentro il quadrato) sia la regione scura (fuori) — e perché il vertice è un angolo (non un bordo dritto), la porzione del cerchio dentro la zona chiara sarà un **arco contiguo** di dimensione significativa, non sparsa a caso.

Se invece $p$ fosse su un'area uniforme (tutta chiara o tutta scura), tutti i 16 pixel del cerchio sarebbero "simili" al centro — nessun arco di 12 chiari/scuri, quindi non viene classificato come angolo. Se $p$ fosse su un bordo dritto (non un angolo), il cerchio sarebbe diviso in due archi di circa 8 pixel ciascuno (metà chiaro, metà scuro) — non raggiunge la soglia di 12 contigui, quindi nemmeno questo viene classificato come angolo. Solo in un vero angolo (dove la geometria locale "piega" bruscamente) si ottiene un arco abbastanza grande e contiguo.

### 3.5 La scorciatoia per andare veloce

Controllare tutti i 16 pixel per *ogni* pixel dell'immagine sarebbe lento. FAST usa un trucco: prima controlla solo 4 pixel "cardinali" del cerchio (a 0°, 90°, 180°, 270° — pensa alle ore 12, 3, 6, 9 su un orologio). Se meno di 3 di questi 4 soddisfano già la condizione "più chiaro/più scuro", il pixel viene **scartato immediatamente** — non serve controllare gli altri 12. Dato che la maggior parte dei pixel di un'immagine *non* sono angoli, questo scarta velocemente la stragrande maggioranza dei candidati con pochissimi confronti, ed è il motivo per cui FAST è... fast.

---

## 4. Rendere il punto riconoscibile da angolazioni diverse: l'orientamento

### 4.1 Il problema nuovo

Trovare *dove* sono gli angoli (FAST) è solo metà del lavoro. Ora dobbiamo **descrivere** ogni angolo in modo che, se lo stesso punto fisico appare ruotato in un altro frame (perché hai ruotato la camera o il target), il sistema lo riconosca comunque come "lo stesso punto".

### 4.2 Cosa significa "il centroide di intensità" di una piccola zona

Prendi una piccola patch quadrata di pixel attorno al keypoint trovato da FAST. Ogni pixel ha una posizione $(x,y)$ relativa al centro della patch, e un'intensità $I(x,y)$.

Pensa all'intensità come a un "peso": pixel più chiari pesano di più. Il **centroide pesato** della patch è come il baricentro di un oggetto fisico dove la "massa" in ogni punto è la luminosità lì:

$$
C_x = \frac{\sum_{x,y} x \cdot I(x,y)}{\sum_{x,y} I(x,y)}, \qquad C_y = \frac{\sum_{x,y} y \cdot I(x,y)}{\sum_{x,y} I(x,y)}
$$

(In notazione compatta della letteratura, questi rapporti si scrivono con i "momenti" $m_{10}, m_{01}, m_{00}$ — sono solo nomi per queste stesse somme.)

### 4.3 Perché il centroide ci dà una direzione utile

Se la patch ha intensità distribuita simmetricamente attorno al centro geometrico, il centroide pesato coincide (circa) con il centro — nessuna direzione preferita. Ma se la patch ha, per esempio, una zona chiara concentrata in alto a destra (tipico vicino a un angolo, dove c'è un bordo chiaro-scuro inclinato), il centroide pesato si **sposta** verso quella zona.

Il vettore dal centro geometrico della patch al centroide pesato definisce quindi una **direzione dominante locale** — un orientamento $\theta$ intrinseco a quella specifica zona dell'immagine:

$$
\theta = \arctan2(C_y, C_x)
$$

($\arctan2$ è l'arcotangente "a due argomenti" che già conosci dalla trigonometria — restituisce l'angolo corretto in tutti i 4 quadranti, non solo in $[-\pi/2, \pi/2]$ come l'arctan semplice).

### 4.4 Come questo risolve il problema della rotazione

Ecco il punto cruciale: se l'intera immagine ruota (camera o target che ruota), **anche il pattern di intensità nella patch ruota della stessa quantità**, e quindi **anche $\theta$ ruota della stessa quantità**. $\theta$ "segue" la rotazione dell'immagine.

Questo significa che possiamo usare $\theta$ per **compensare** la rotazione prima di descrivere il punto: ruotiamo virtualmente la patch di $-\theta$ (riportandola al suo "orientamento canonico"), e *poi* calcoliamo il descriptor (prossima sezione) su questa versione raddrizzata. Se la stessa patch fisica viene vista ruotata in un altro frame, il suo $-\theta$ sarà diverso, ma il risultato dopo la compensazione sarà **lo stesso** descriptor (a meno di rumore) — questo è ciò che si intende per "rotation invariant".

---

## 5. Descrivere un punto con dei bit: BRIEF

### 5.1 L'idea, molto semplice

Vogliamo trasformare l'aspetto di una piccola patch di pixel in una sequenza di numeri che possiamo confrontare velocemente. BRIEF fa questo nel modo più semplice possibile: **confronti binari tra coppie di pixel**.

Si scelgono in anticipo (una volta, non per ogni immagine) un insieme di $n$ coppie di posizioni relative al centro della patch, es. $(p_1, q_1), (p_2, q_2), \ldots, (p_n, q_n)$ — tipicamente $n = 256$. Queste posizioni sono fisse, decise a priori (in pratica scelte con un certo criterio statistico, ma per capire il concetto puoi pensarle come "fisse e date").

### 5.2 Il confronto

Per ogni coppia $i$, guardiamo i due pixel nella patch corrente e produciamo **un singolo bit**:

$$
\tau_i = \begin{cases} 1 & \text{se } I(p_i) < I(q_i) \\ 0 & \text{se } I(p_i) \geq I(q_i) \end{cases}
$$

In parole: "il primo pixel della coppia è più scuro del secondo?" Sì → 1, No → 0.

### 5.3 Il descriptor finale

Concatenando i 256 bit ottenuti da tutte le coppie, otteniamo una stringa di 256 bit (32 byte) che descrive quella patch:

$$
\text{descriptor} = \tau_1 \tau_2 \tau_3 \ldots \tau_{256}
$$

### 5.4 Perché funziona, anche se ogni singolo bit è "debole"

Un singolo confronto ($I(p_1) < I(q_1)$?) porta pochissima informazione — è quasi una moneta lanciata, potrebbe essere vero per molte patch diverse. Ma **256 confronti indipendenti insieme** sono molto più informativi: è come chiedere "sì/no" a 256 domande diverse sulla stessa patch — la combinazione specifica di risposte diventa una "firma" che, con alta probabilità, è diversa da quella di una patch presa da un punto diverso dell'immagine.

### 5.5 ORB = FAST (dove) + orientamento (sezione 4) + BRIEF ruotato (qui)

Il nome "ORB" sta per **O**riented FAST and **R**otated **B**RIEF. Metti insieme i pezzi:
- **FAST** trova *dove* sono i keypoint (sezione 3)
- Il **centroide di intensità** calcola un orientamento $\theta$ per ogni keypoint (sezione 4)
- **BRIEF** viene applicato non sulla patch originale, ma su una versione virtualmente **ruotata di $-\theta$** — le posizioni delle coppie $(p_i, q_i)$ vengono anch'esse ruotate di $\theta$ prima di leggere i pixel:

$$
\begin{pmatrix} p_i' \\ q_i' \end{pmatrix} = R_\theta \begin{pmatrix} p_i \\ q_i \end{pmatrix}, \qquad R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

(questa è la matrice di rotazione 2D standard, identica a quella che probabilmente hai già visto in grafica/Unity per ruotare oggetti nel piano)

Il risultato finale è un descriptor a 256 bit che è (parzialmente) invariante alla rotazione dell'immagine.

---

## 6. Confrontare due descriptor binari: la distanza di Hamming

### 6.1 Il problema

Abbiamo un descriptor a 256 bit per ogni keypoint del target, e uno per ogni keypoint del frame. Vogliamo sapere: *quanto sono simili* due descriptor, per decidere se rappresentano lo stesso punto fisico.

### 6.2 Perché non la distanza euclidea (quella "classica" tra vettori)

Se i descriptor fossero vettori di numeri continui (come per SIFT), la distanza euclidea classica ($\sqrt{\sum (a_i - b_i)^2}$) avrebbe senso: misura quanto sono "vicini" geometricamente in uno spazio continuo.

Ma qui abbiamo **bit** prodotti da confronti booleani indipendenti. Il concetto naturale di "differenza" non è una distanza geometrica continua — è semplicemente: **in quante delle 256 domande sì/no le risposte sono diverse?**

### 6.3 La distanza di Hamming

$$
d_H(A, B) = \sum_{i=1}^{256} (A_i \oplus B_i)
$$

dove $\oplus$ è lo XOR (or esclusivo): vale 1 se i due bit sono diversi, 0 se sono uguali. La somma totale conta quante posizioni differiscono.

**Esempio concreto con stringhe più corte:** se $A = 1011$ e $B = 1001$, confrontando bit per bit: 1=1 (uguale), 0=0 (uguale), 1≠0 (diverso), 1=1 (uguale) → distanza di Hamming = 1 (un solo bit differisce).

### 6.4 Perché è anche più veloce da calcolare

Un vantaggio pratico enorme: l'operazione XOR tra due numeri binari è una singola istruzione hardware nel processore, e contare i bit a 1 risultanti (popcount) è anch'essa accelerata in hardware su CPU moderne. Confrontare due descriptor da 256 bit con Hamming è quindi **molto più veloce** di calcolare una distanza euclidea tra vettori float — altro motivo per cui ORB+Hamming è la scelta giusta quando serve velocità (il tracking in tempo reale che ti serve per l'AR).

---

## 7. La Homography: come si trasforma un piano visto da angoli diversi

### 7.1 Il problema specifico

Il target è **piatto** (un piano nel mondo reale: una foto, un poster). Quando lo guardi da angolazioni diverse, la sua immagine nel frame si deforma (si "schiaccia" prospetticamente) — ma è sempre la stessa trasformazione geometrica per *tutti* i punti di quel piano, perché stanno tutti sulla stessa superficie 2D.

Questa trasformazione specifica — che mappa un piano 2D in un altro piano 2D tenendo conto della prospettiva — si chiama **homography**.

### 7.2 La homography come matrice

Si rappresenta con una matrice 3×3 (usiamo di nuovo le coordinate omogenee della sezione 2.5: un punto $(x,y)$ del target diventa $(x,y,1)$):

$$
\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}, \qquad H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix}
$$

Le coordinate finali nel frame si ottengono dividendo per l'ultima componente: $x_{\text{frame}} = x'/w'$, $y_{\text{frame}} = y'/w'$ — esattamente lo stesso trucco della sezione 2.5 per gestire la non-linearità della prospettiva.

### 7.3 Perché 8 numeri liberi, non 9

$H$ ha 9 elementi, ma c'è una sottigliezza: se moltiplichi *tutta* la matrice $H$ per una costante qualsiasi (es. raddoppi ogni elemento), la trasformazione risultante **non cambia** — perché poi dividi per $w'$, e quella divisione "annulla" qualsiasi fattore di scala comune. Quindi $H$ è definita "a meno di scala": di fatto solo i **rapporti** tra i 9 elementi contano, il che equivale a dire che ci sono **8 gradi di libertà** indipendenti.

### 7.4 Quanti punti corrispondenti servono per trovare H?

Ogni coppia di punti corrispondenti (un punto sul target, e dove appare nel frame) dà informazione su $H$. Vediamo quanta, esattamente.

Partiamo dal fatto che $\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix}$ e $H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$ devono essere **paralleli** (puntano nella stessa direzione, anche se di lunghezza diversa — perché la "scala" $w'$ non è fissata a priori). Due vettori paralleli hanno **prodotto vettoriale nullo**. Questo prodotto vettoriale nullo, scritto per componenti, dà — dopo un po' di algebra — **2 equazioni lineari indipendenti** nei coefficienti $h_{11} \ldots h_{33}$, per ogni coppia di punti.

Con 8 incognite indipendenti (vedi 7.3) e 2 equazioni per ogni coppia di punti, **servono almeno 4 coppie** ($4 \times 2 = 8$ equazioni) per determinare $H$ univocamente.

### 7.5 Cosa succede con più di 4 punti (il nostro caso reale)

Nel tuo script hai decine di match, non solo 4. Con più equazioni che incognite, il sistema è **sovra-determinato**: non esiste necessariamente una $H$ che soddisfi *esattamente* tutte le equazioni (perché i match reali hanno rumore — piccoli errori di posizione, qualche outlier).

La soluzione pratica è trovare la $H$ che minimizza l'errore complessivo nel senso dei **minimi quadrati** — esattamente lo stesso principio che probabilmente hai visto in altri contesti (es. fit di una retta a punti sparsi: non passa esattamente per tutti, minimizza la somma degli errori). La tecnica specifica usata si chiama SVD (Singular Value Decomposition, decomposizione ai valori singolari) — non serve che tu sappia implementarla, ma il concetto chiave da portare a casa è: **`cv2.findHomography` sta risolvendo un problema di minimi quadrati**, proprio come quando fitti una curva a dei dati sperimentali.

---

## 8. RANSAC: come ignorare i match sbagliati

### 8.1 Il problema con i minimi quadrati "puri"

I minimi quadrati della sezione 7.5 trattano *tutti* i punti allo stesso modo, cercando di minimizzare l'errore medio. Il problema: se anche solo pochi dei tuoi match sono **completamente sbagliati** (outlier — falsi positivi del matching ORB), questi possono "tirare" la soluzione lontano dalla vera trasformazione, perché il minimo quadrato cerca di accontentare anche loro.

**Esempio intuitivo:** se stai facendo la media di 10 misure e una di queste è un errore enorme (es. hai letto male lo strumento), quella singola misura sbagliata sposta significativamente la media — anche se le altre 9 erano accurate.

### 8.2 L'idea di RANSAC (RANdom SAmple Consensus)

Invece di usare *tutti* i punti insieme fin da subito, RANSAC fa una scommessa intelligente, ripetuta molte volte:

1. Scegli a caso un **piccolo sottoinsieme** di punti — il minimo necessario, cioè 4 (sezione 7.4)
2. Calcola la $H$ che passa esattamente per questi 4 punti
3. Controlla: **quanti altri punti** (tra tutti quelli disponibili) sono coerenti con questa $H$? ("coerenti" = se proietti il punto del target con questa $H$, cade vicino — sotto una soglia in pixel — alla sua posizione osservata nel frame)
4. Ripeti i passi 1-3 molte volte, con sottoinsiemi diversi scelti a caso
5. Alla fine, tieni la $H$ che ha ottenuto il **maggior numero di punti coerenti** (questi punti coerenti si chiamano **inliers**, gli altri **outlier**)

### 8.3 Perché funziona

L'intuizione: se scegli a caso 4 punti che sono *tutti* outlier (sbagliati), la $H$ che ne risulta sarà casuale e **pochi altri punti** saranno coerenti con essa. Ma se per fortuna scegli 4 punti che sono *tutti* inlier (corretti), la $H$ risultante sarà quella vera, e **molti altri punti corretti** risulteranno coerenti. Ripetendo tante volte, prima o poi capiti su un campione "tutto corretto", e quel tentativo si distingue chiaramente dagli altri per il numero di punti coerenti che produce.

### 8.4 Quante volte devo ripetere? Un calcolo di probabilità

Questa è la parte con un po' di matematica, ma è solo probabilità elementare che già conosci.

Sia $w$ la frazione di match che sono effettivamente corretti (inlier) — es. se metà dei tuoi match sono giusti, $w = 0.5$. La probabilità che, scegliendo 4 punti **a caso**, capitino **tutti e 4** inlier è (assumendo scelte indipendenti, una semplificazione comune):

$$
P(\text{campione tutto inlier}) = w \times w \times w \times w = w^4
$$

Quindi la probabilità che un singolo tentativo **fallisca** (almeno uno dei 4 sia un outlier) è:

$$
P(\text{fallimento singolo tentativo}) = 1 - w^4
$$

Se ripeti $k$ tentativi **indipendenti**, la probabilità che **falliscano tutti e $k$** (mai un campione fortunato tutto-inlier) è:

$$
P(\text{fallimento totale dopo } k \text{ tentativi}) = (1 - w^4)^k
$$

Vogliamo che questa probabilità di fallimento totale sia molto piccola — diciamo, vogliamo essere sicuri al 99% di aver trovato almeno un campione buono, quindi vogliamo che la probabilità di fallimento sia $\leq 0.01$:

$$
(1 - w^4)^k \leq 0.01
$$

Per isolare $k$, prendiamo il logaritmo di entrambi i lati (puoi usare qualsiasi base, qui naturale). Nota: quando si applica il logaritmo a una disuguaglianza con basi minori di 1 (qui $1-w^4 < 1$), il logaritmo è negativo, e bisogna invertire il segno della disuguaglianza dividendo per un numero negativo:

$$
k \cdot \log(1 - w^4) \leq \log(0.01)
$$

$$
k \geq \frac{\log(0.01)}{\log(1 - w^4)}
$$

(la disuguaglianza si inverte perché $\log(1-w^4)$ è negativo, dividere per un numero negativo inverte il verso)

### 8.5 Mettiamo un numero vero

Supponiamo $w = 0.5$ (metà dei match sono outlier — uno scenario abbastanza rumoroso). Calcoliamo:

$$
1 - w^4 = 1 - 0.5^4 = 1 - 0.0625 = 0.9375
$$

$$
k \geq \frac{\log(0.01)}{\log(0.9375)} = \frac{-4.605}{-0.0645} \approx 71.4
$$

Quindi **circa 71-72 tentativi** bastano per essere sicuri al 99% di trovare la $H$ corretta, anche con metà dei dati sbagliati. Questo è il punto chiave da capire: il numero di iterazioni necessarie **non esplode** anche con tanto rumore — cresce in modo relativamente contenuto, motivo per cui RANSAC è praticabile in tempo reale.

### 8.6 Collegamento diretto al codice

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

Il numero `5.0` è la soglia (in pixel) usata nel passo 3 della sezione 8.2 — quanto "vicino" deve cadere un punto proiettato per essere considerato coerente (inlier). `mask` è un array di 0 e 1 che ti dice, per ogni punto che hai passato, se è stato classificato come inlier (1) o outlier (0) dalla $H$ vincente — è esattamente il numero che hai visto stampato in console come "inliers".

---

## 9. Dalla Homography alla Posa 3D vera: il problema PnP

### 9.1 Perché la homography non è ancora abbastanza

La homography ti dice: "il piano del target, visto da questo frame, è deformato così" — è una trasformazione **2D che diventa 2D**. Ma per disegnare un cubo che si "alza" *fuori* dal piano del target (non solo un contorno piatto sopra), serve sapere **dove sta realmente la camera nello spazio 3D** — la rotazione $R$ e traslazione $t$ vere, viste nella sezione 2.6.

### 9.2 Il problema PnP, formulato per bene

"PnP" sta per "**P**erspective-**n**-**P**oint": dati $n$ punti di cui conosci sia la posizione **3D nel mondo** $P_i = (X_i, Y_i, Z_i)$ sia dove **appaiono in pixel** nel frame $p_i = (u_i, v_i)$, trova $R$ e $t$ che spiegano questa osservazione, usando la formula di proiezione che già conosci dalla sezione 2:

$$
\lambda_i \begin{pmatrix} u_i \\ v_i \\ 1 \end{pmatrix} = K \left( R \, P_i + t \right) \quad \text{per ogni punto } i
$$

($\lambda_i$ è un fattore di scala che si "consuma" nella divisione finale, come nella sezione 2.5 — ogni punto ha il suo $\lambda_i$ perché ogni punto può essere a una $Z$ diversa)

### 9.3 Perché è più difficile di trovare la homography

Qui $R$ non è una matrice qualsiasi: deve essere una **vera rotazione**, il che significa che ha vincoli precisi (le sue colonne devono essere perpendicolari tra loro e di lunghezza 1 — i vincoli di "ortonormalità" che probabilmente hai visto in algebra lineare). Questo rende il problema **non lineare**, a differenza della homography (sezione 7) che era un sistema lineare risolvibile con i minimi quadrati standard.

### 9.4 Il caso speciale: target piatto (il nostro caso esatto)

Buona notizia: nel nostro caso il target è un piano, quindi tutti i punti 3D che usiamo hanno $Z = 0$ (li definiamo nel sistema di coordinate del target stesso, dove il piano del target *è* il piano $Z=0$). Quando $Z_i = 0$ per tutti i punti, la formula di proiezione si semplifica: nella moltiplicazione $R \, P_i$, la **terza colonna di $R$** viene sempre moltiplicata per $Z_i = 0$ e quindi "scompare" dal calcolo. Restano solo le prime due colonne di $R$ (chiamiamole $r_1, r_2$):

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \begin{pmatrix} r_1 & r_2 & t \end{pmatrix} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}
$$

**Osservazione chiave:** la matrice $K (r_1 \; r_2 \; t)$ è una matrice 3×3 che moltiplica un punto omogeneo 2D $(X,Y,1)$ per dare un punto omogeneo 2D $(u,v,1)$ (a meno di scala $\lambda$) — questa è **esattamente la struttura di una homography** (sezione 7.2)! In altre parole: quando il target è piatto, homography e posa 3D sono **due descrizioni della stessa informazione geometrica**, solo che la homography non impone il vincolo che $r_1, r_2$ siano davvero parte di una matrice di rotazione valida, mentre `solvePnP` lo impone esplicitamente.

Questo è il ponte concettuale diretto tra quello che hai già calcolato nello step 3 (la homography) e quello che `solvePnP` fa nello step 4: `solvePnP` può usare la homography come **punto di partenza**, e poi "raddrizzarla" matematicamente per estrarre una vera rotazione $R$ e traslazione $t$ coerenti.

### 9.5 Come si risolve in pratica (a grandi linee)

Una volta che si ha una stima iniziale di $R, t$ (es. dalla relazione con la homography sopra), OpenCV raffina la soluzione minimizzando l'**errore di riproiezione**: per la posa corrente $(R,t)$, calcola dove *dovrebbero* proiettarsi i punti 3D noti, e confronta con dove sono *effettivamente osservati* nel frame:

$$
\text{errore} = \sum_{i=1}^{n} \left\| p_i^{\text{osservato}} - p_i^{\text{previsto}}(R, t) \right\|^2
$$

Poi aggiusta iterativamente $R, t$ per ridurre questo errore (un algoritmo di ottimizzazione iterativa chiamato Levenberg-Marquardt, concettualmente simile a una discesa del gradiente che probabilmente hai incontrato in contesti di machine learning: muoviti nella direzione che riduce l'errore, ripeti finché l'errore non si stabilizza).

---

## 10. Rappresentare una rotazione con solo 3 numeri: Rodrigues

### 10.1 Il problema di ridondanza

Una matrice di rotazione $R$ ha 9 numeri (3×3), ma sappiamo che una rotazione nello spazio 3D ha solo **3 gradi di libertà reali** (puoi pensarlo come: 1 asse attorno a cui ruotare, definito da 2 angoli per la direzione, più 1 angolo di quanto ruotare — totale 3 numeri indipendenti). I restanti 6 numeri della matrice sono ridondanti, vincolati dalle condizioni di ortonormalità.

`solvePnP` restituisce `rvec`, un vettore di **soli 3 numeri**, che codifica la stessa informazione di $R$ in modo compatto.

### 10.2 Cosa rappresenta esattamente rvec

Un vettore di rotazione $\mathbf{r}$ codifica **due informazioni in uno**:
- La sua **direzione** (normalizzata, lunghezza 1) è l'**asse** attorno a cui avviene la rotazione — pensa a uno spiedo che attraversa l'oggetto, attorno a cui l'oggetto ruota
- La sua **lunghezza** $\theta = \|\mathbf{r}\|$ è **l'angolo** di rotazione (in radianti) attorno a quell'asse

Quindi $\mathbf{r} = \theta \hat{n}$, dove $\hat{n}$ è il versore (vettore di lunghezza 1) che indica l'asse.

### 10.3 La formula di Rodrigues: come passare da rvec a R

Per ricostruire la matrice di rotazione $R$ completa a partire da $\theta$ e $\hat{n}$, si usa:

$$
R = I + \sin\theta \, [\hat{n}]_\times + (1 - \cos\theta) \, [\hat{n}]_\times^2
$$

Qui $I$ è la matrice identità, e $[\hat{n}]_\times$ è una matrice speciale (3×3, antisimmetrica) costruita dalle componenti di $\hat{n} = (n_x, n_y, n_z)$:

$$
[\hat{n}]_\times = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}
$$

Questa matrice ha una proprietà speciale: moltiplicarla per un vettore $\mathbf{v}$ qualsiasi dà lo stesso risultato del prodotto vettoriale $\hat{n} \times \mathbf{v}$ — è solo un modo di scrivere il prodotto vettoriale come moltiplicazione matriciale (probabilmente l'hai già visto in qualche corso di fisica/grafica per calcolare momenti o torsioni).

**Non serve che tu memorizzi questa formula** — il punto concettuale importante è: *un vettore a 3 componenti contiene tutta l'informazione necessaria per ricostruire una rotazione completa*, e questa formula è semplicemente la "ricetta" per farlo.

### 10.4 Perché questo importa per il bug che avevi trovato

Ricordi il bug del reset mancante nello smoothing? La causa profonda di quel problema, a livello più teorico, è questa: **mediare due vettori `rvec` con una somma pesata semplice (EMA) non è geometricamente lo stesso che "mediare due rotazioni"**.

Perché: lo spazio dei vettori di rotazione non è uno spazio vettoriale "ben comportato" per fare medie dirette, soprattutto quando l'angolo $\theta$ è grande o quando ci sono ambiguità (la stessa rotazione fisica può essere rappresentata da più di un vettore `rvec`, ad esempio ruotare di $\theta$ attorno a $\hat{n}$, oppure di $-\theta$ attorno a $-\hat{n}$, danno la stessa rotazione finale — ma sono vettori diversi numericamente). Mediare due `rvec` "diversi sulla carta" ma che rappresentano rotazioni vicine può, in casi specifici, dare un risultato che non corrisponde a nessuna rotazione sensata nel mezzo.

**La soluzione matematicamente corretta** userebbe i **quaternioni** (un'altra rappresentazione delle rotazioni, a 4 componenti, con proprietà algebriche più "comode" per l'interpolazione) e una tecnica chiamata **SLERP** (Spherical Linear intERPolation), che interpola lungo il percorso più corto sulla sfera 4-dimensionale dei quaternioni unitari, garantendo che ogni passo intermedio sia sempre una rotazione valida.

**Perché nel tuo caso "funziona comunque bene":** quando le rotazioni frame-to-frame sono **piccole** (il target non si muove a scatti enormi tra un frame e il successivo, che è il caso tipico di un tracking fluido), l'approssimazione lineare dell'EMA resta numericamente molto vicina al risultato corretto — l'errore introdotto è piccolo. Diventa un problema visibile principalmente quando il tracking si "rompe" per qualche frame e poi riprende con una posa molto diversa da quella vecchia "congelata" — esattamente lo scenario del bug che hai trovato e corretto con il reset.

---

## 11. Mappa Concettuale Finale

```
Frame webcam (matrice di pixel)
       │
       ▼
[Sezione 3] FAST: per ogni pixel, controlla 16 vicini su un cerchio
            → trova pixel "angolo" (keypoints)
       │
       ▼
[Sezione 4] Centroide di intensità nella patch attorno al keypoint
            → orientamento θ specifico di quel punto
       │
       ▼
[Sezione 5] BRIEF: 256 confronti binari tra coppie di pixel,
            ruotate di θ per compensare l'orientamento
            → descriptor a 256 bit (ORB completo = FAST+orient.+BRIEF ruotato)
       │
       ▼
[Sezione 6] Distanza di Hamming tra descriptor del target e del frame
            → coppie di punti che sembrano corrispondersi (matches)
       │
       ▼
[Sezione 8] RANSAC: prova sottoinsiemi casuali di 4 punti,
            tieni la trasformazione con più punti coerenti
       │
       ▼
[Sezione 7] Homography H: la trasformazione di piano vincente
            (calcolata internamente con minimi quadrati/SVD)
       │
       ▼
[Sezione 9] H, essendo strutturalmente equivalente a K[r1 r2 t]
            per un piano, fornisce la stima iniziale per solvePnP
            → solvePnP raffina e restituisce R (vera rotazione) e t
       │
       ▼
[Sezione 10] R si rappresenta in modo compatto come rvec (Rodrigues)
       │
       ▼
K [R|t] applicato ai vertici del modello 3D (projectPoints)
       │
       ▼
Vertici del cubo/piramide proiettati in pixel → disegnati sul frame
```

---

## 12. Glossario Rapido (per quando rileggi velocemente)

| Termine | Cos'è, in una frase |
|---|---|
| Keypoint | Un punto dell'immagine giudicato "distintivo" (tipicamente un angolo) |
| Descriptor | Una stringa di numeri (qui: 256 bit) che descrive l'aspetto locale attorno a un keypoint |
| Matching | Il processo di trovare, per ogni keypoint del target, il keypoint più simile nel frame |
| Inlier | Un match che risulta coerente con la trasformazione geometrica trovata (target reale) |
| Outlier | Un match sbagliato, scartato da RANSAC |
| Homography | Matrice 3×3 che descrive come un piano si trasforma in un altro piano (es. per prospettiva) |
| Posa (pose) | Rotazione + traslazione: descrive dove sta e come è orientata la camera nello spazio |
| rvec / tvec | Rappresentazione compatta (3 numeri ciascuno) di rotazione e traslazione |
| Riproiezione | Il processo di "disegnare di nuovo" un punto 3D noto usando la posa stimata, per verificare quanto è accurata |

---

*Documento di studio personale — non destinato a pubblicazione esterna.*
