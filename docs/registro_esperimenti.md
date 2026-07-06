# Registro degli esperimenti

Questo documento elenca soltanto esperimenti realmente eseguiti. I risultati
quantitativi vanno considerati definitivi solo dopo la revisione della ground
truth indicata nelle note.

## E01 — SAM2 con box manuale

- **Data:** 2026-07-03; ripetizione e valutazione v2 il 2026-07-04
- **Stato:** E01-v2 completato e archiviato
- **Video:** `test_video_san_donato.mp4`
- **Intervallo:** 85–105 secondi
- **Frame:** 500, risoluzione 640×512, circa 25 fps
- **Modello:** SAM 2.1 Small, `sam2.1_hiera_small.pt`
- **Hardware:** Google Colab, NVIDIA Tesla T4
- **Fine-tuning:** no
- **Prompt:** box al frame iniziale, coordinate `[392, 195, 438, 270]`
- **Oggetto:** persona

### Osservazione qualitativa

SAM2 mantiene l'identità della persona nei frame campionati 0, 100, 200, 300,
400 e 499, anche in presenza di altre persone e cambiamenti di postura. È
presente un piccolo componente spurio disconnesso nel primo frame.

### Valutazione box v1 storica

Confronto sulla track CVAT `1`, frame sorgente comuni 2135–2624:

| Soglia IoU | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0,3 | 408 | 82 | 82 | 0,833 | 0,833 | 0,833 |
| 0,5 | 350 | 140 | 140 | 0,714 | 0,714 | 0,714 |

IoU media `0,587`; IoU mediana `0,662`.

### Limite della valutazione v1

Tra circa 98 e 101 secondi la box CVAT interpolata è spostata rispetto alla
persona, mentre la maschera SAM2 appare visivamente corretta. Queste metriche
non vanno presentate come definitive finché la ground truth non viene
revisionata.

### Valutazione E01-v2

L'esperimento è stato ripetuto con lo stesso prompt e valutato sulla ground
truth San Donato v2, track `0`, nei 490 frame sorgente comuni 2135–2624.

| Soglia IoU | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0,3 | 490 | 0 | 0 | 1,000 | 1,000 | 1,000 |
| 0,5 | 490 | 0 | 0 | 1,000 | 1,000 | 1,000 |

IoU media `0,755`; mediana `0,769`; minima `0,561`; massima `0,981`.
I primi 10 frame della clip sono esclusi perché la track CVAT v2 inizia al
frame sorgente 2135.

Il risultato misura il tracking condizionato da una box manuale accurata:
non rappresenta anomaly detection autonoma.

### Artefatti

- configurazione completa: `results/qualitative/sam2_pilot_01/experiment.json`;
- video overlay: `results/qualitative/sam2_pilot_01/sam2_san_donato_85_105_h264.mp4`;
- maschere e box: `results/qualitative/sam2_pilot_01/colab_export/`;
- metriche per frame: `results/metrics/sam2_pilot_01/per_frame_iou.csv`;
- riepilogo: `results/metrics/sam2_pilot_01/summary.csv`;
- notebook: `notebooks/sam2_pilot_colab.ipynb`.

Artefatti E01-v2:

- pacchetto completo: `results/qualitative/sam2_pilot_01_v2/E01_v2_sam2_manual_box_results.zip`;
- video, maschere, predizioni e metadati:
  `results/qualitative/sam2_pilot_01_v2/`;
- metriche aggregate e per frame:
  `results/metrics/sam2_pilot_01_v2/`;
- notebook Colab eseguito: `notebooks/E01_SAM2_v2.ipynb`;
- SHA256 notebook:
  `ae890896cfe2136789f9a0fd654c476aad4cb4c67a8ed1a9838ec884e933834c`;
- commit SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

## E02 — Grounding DINO + SAM2

- **Data:** 2026-07-03; estensione periodica e valutazione v2 il 2026-07-04
- **Stato:** E02-v2 completato e archiviato; E02-v1 conservato come storico
- **Video e intervallo:** come E01, San Donato 85–105 secondi
- **Prompt testuale:** `person`
- **Detector:** `IDEA-Research/grounding-dino-tiny`
- **Soglie:** detection 0,20; testo 0,15; selezione 0,30
- **Periodicità E02-v2:** una detection ogni 50 frame
- **Tracker:** SAM 2.1 Small
- **Fine-tuning:** no

### Risultato iniziale E02-v1

Grounding DINO produce tre box:

- persona al bordo sinistro: score 0,771;
- persona centrale: score 0,631;
- regione debole sullo sfondo: score 0,22.

La soglia 0,30 conserva le due persone reali ed elimina la regione debole. Le
due box vengono passate automaticamente a SAM2, che mantiene entrambe le
identità per 500 frame.

### Confronto E02-v1 storico

| Oggetto | Track CVAT | Frame comuni | IoU media | Frame IoU≥0,3 | Frame IoU≥0,5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 227 | 0,542 | 227 | 133 |
| 2 | 1 | 490 | 0,579 | 402 | 349 |

La persona 1 è visibile per 273 frame prima che inizi la track CVAT 3. Questo
conferma che le metriche non possono ancora essere trattate come definitive.

### Limite E02-v1 e correzione E02-v2

Una terza persona entra dopo il frame iniziale e non viene rilevata: il
detector testuale era stato eseguito soltanto al frame 0. In E02-v2 Grounding
DINO viene eseguito ogni 50 frame. La nuova persona è rilevata al frame clip
250, sorgente 2375, con score `0,526` e aggiunta a SAM2 come ID `3`.

Al frame clip 350 il detector produce quattro proposte ma soltanto tre persone
distinte: una proposta è duplicata e viene respinta dal confronto con le
track già attive. Questo caso mostra perché il numero di box grezze non deve
essere interpretato direttamente come numero di persone.

### Valutazione E02-v2

Le box ricavate dalle maschere SAM2 sono confrontate con le track CVAT v2 di
tipo `persona`. Le track `incerto` sono escluse. Le associazioni verificate
sono ID `1` → track `10`, ID `2` → track `0` e ID `3` → track `12`.

| ID | Track CVAT | Frame comuni | IoU media | F1 IoU≥0,3 | F1 IoU≥0,5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 500 | 0,675 | 1,000 | 0,888 |
| 2 | 0 | 490 | 0,751 | 1,000 | 1,000 |
| 3 | 12 | 242 | 0,684 | 1,000 | 0,959 |
| **Aggregato** | 0, 10, 12 | 1232 | 0,707 | 1,000 | 0,946 |

Non risultano frame GT mancanti nelle tre track abbinate. L'ID 2 precede di
10 frame l'inizio della track CVAT e l'ID 3 di 8 frame; questi frame extra
sono riportati separatamente e non entrano nella metrica sull'intervallo
comune. Vicino al frame finale gli ID 1 e 3 sono fortemente sovrapposti e la
maschera dell'ID 3 si restringe.

Il risultato misura il rilevamento della classe semantica `person` e il
tracking condizionato dalle box di Grounding DINO. Non costituisce anomaly
detection generale.

### Artefatti

- configurazione: `results/qualitative/grounded_sam2_pilot_02/experiment.json`;
- video e figure: `results/qualitative/grounded_sam2_pilot_02/`;
- box, maschere e detection: sottocartella `colab_export/`;
- associazione alle track CVAT:
  `results/metrics/grounded_sam2_pilot_02/candidate_track_matching.csv`.

Artefatti E02-v2:

- pacchetto completo:
  `results/qualitative/grounded_sam2_pilot_02_v2/E02_v2_grounding_dino_periodic_sam2_results.zip`;
- video, figure, detection, prompt e maschere compresse:
  `results/qualitative/grounded_sam2_pilot_02_v2/colab_export/`;
- metriche aggregate e per frame:
  `results/metrics/grounded_sam2_pilot_02_v2/`;
- notebook Colab eseguito: `notebooks/E02_Grounded_SAM2_v2.ipynb`;
- SHA256 notebook:
  `5622417c5f1d802ef27fffef708498b4391f7029f00bb5240d681716164607f9`;
- SHA256 del pacchetto:
  `b1a3c42b1e74315d9126eb35880052ca42f21a7b832eafcc0149d5ba4ac43f3c`.

## E03 — Sensibilità di Grounding DINO al prompt

- **Data:** 2026-07-03
- **Stato:** pilot completato del modulo ausiliario di proposta; non è ancora
  un esperimento completo di anomaly detection
- **Video:** `test_video_san_donato.mp4`
- **Frame sorgente:** 2125, corrispondente al frame iniziale della clip E01/E02
- **Detector:** `IDEA-Research/grounding-dino-tiny`
- **Prompt:** `person`, `box`, `pallet`, `obstacle`, `foreign object`,
  `anomaly`
- **Soglie:** detection 0,20; testo 0,15; selezione 0,30
- **Hardware:** Google Colab, NVIDIA Tesla T4
- **Fine-tuning:** no

### Risultati

| Prompt | Rilevamenti ≥0,20 | Selezionati ≥0,30 | Score massimo |
|---|---:|---:|---:|
| `person` | 3 | 2 | 0,771 |
| `box` | 3 | 3 | 0,478 |
| `pallet` | 6 | 4 | 0,487 |
| `obstacle` | 7 | 5 | 0,508 |
| `foreign object` | 8 | 6 | 0,590 |
| `anomaly` | 6 | 2 | 0,558 |

`person` conserva le due persone reali ed elimina una regione debole. I prompt
generici `obstacle` e `foreign object` aumentano il numero di candidati, ma
producono regioni eterogenee. `anomaly` seleziona la persona centrale e un
oggetto, ma non la persona al bordo sinistro con la soglia 0,30.

Il tempo maggiore del primo prompt (`person`, 3,09 s) include il warm-up del
modello; i prompt successivi richiedono circa 0,38 s ciascuno.

### Conclusione e limite

Il risultato descrive soltanto come Grounding DINO propone regioni candidate.
Non dimostra che il sistema rilevi anomalie: una classe riconosciuta non è
automaticamente anomala. Il modulo finale dovrà assegnare un'anomaly score
rispetto alla normalità e al contesto ferroviario. Il confronto riguarda
inoltre un solo frame e non esiste una ground truth per ciascuna classe: il
numero di box non rappresenta precision, recall o accuratezza.

### Artefatti

- figura comparativa:
  `results/qualitative/grounding_dino_prompt_sensitivity_e03/prompt_comparison.png`;
- detection complete e riepilogo: stessa cartella, file CSV;
- configurazione: `metadata.json`;
- archivio originale di Colab: `E03_prompt_sensitivity_results.zip`;
- interpretazione: `analisi.md`.

## E05 — AnomalyDINO 16-shot

- **Data:** 2026-07-05
- **Stato:** completato e archiviato
- **Metodo:** AnomalyDINO, training-free e reference-based
- **Codice AnomalyDINO:** commit
  `b9d1c2648e3a5247437d4d953d907a8f3d994457`
- **Backbone:** DINOv2 ViT-S/14, lato corto 448, griglia patch 32×40
- **Pesi DINOv2:** `dinov2_vits14_pretrain.pth`, SHA256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- **Hardware:** Google Colab, NVIDIA Tesla T4
- **Fine-tuning:** no
- **Preprocessing:** immagini termiche convertite in RGB, senza rotazioni e
  senza mascheramento PCA
- **Ricerca 1-NN:** distanza normalizzata equivalente a `1 - cosine`, calcolo
  esatto tramite moltiplicazione matriciale PyTorch su GPU

### Split congelato

- 16 riferimenti distribuiti uniformemente nei primi 160 frame normali;
- 20 normali successivi per calibrare la soglia;
- 20 normali finali per il test dei falsi positivi;
- 500 frame anomali della clip San Donato 85–105 secondi.

La soglia di frame `0,289022` è il percentile 95 dei soli 20 normali di
calibrazione.

### Classificazione di frame

| TN | FP | FN | TP | Precision | Recall | F1 | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 0 | 500 | 0,992 | 1,000 | 0,996 | 1,000 | 1,000 |

L'inferenza sui 540 frame ha richiesto `44,74 s`, circa `12,07 frame/s`. I
quattro falsi positivi sono i frame normali consecutivi 183–186 e si
concentrano sul ballast a sinistra e sul bordo destro.

### Localizzazione

Il pointing game sulle box di tipo `persona` ottiene 478 successi su 500
frame, accuratezza `0,956`.

| Post-processing | IoU | Precision | Recall | F1 | Frame con almeno un match |
|---|---:|---:|---:|---:|---:|
| soglia assoluta da normali | 0,3 | 0,006 | 0,007 | 0,007 | 26/500 |
| soglia assoluta da normali | 0,5 | 0,000 | 0,000 | 0,000 | 0/500 |
| top 1% adattivo per frame | 0,3 | 0,745 | 0,242 | 0,365 | 500/500 |
| top 1% adattivo per frame | 0,5 | 0,292 | 0,095 | 0,143 | 324/500 |

La soglia spaziale assoluta fallisce perché il cambio globale di scena alza
anche lo sfondo della heatmap. Il top 1% adattivo localizza almeno una
persona in ogni frame a IoU 0,3, ma la griglia patch grossolana non recupera
tutte le 3732 box persona, soprattutto quelle piccole.

### Interpretazione e limiti

La separazione degli score è netta e i massimi cadono spesso sulle persone.
Tuttavia normali e anomalie provengono da video e scene differenti: AUROC e
AP perfette possono riflettere anche domain shift di sfondo, non soltanto gli
oggetti anomali. Inoltre i 500 frame anomali appartengono alla stessa clip e
non sono campioni temporalmente indipendenti. Non sono disponibili maschere
GT pixel-level, quindi non vengono riportate metriche di segmentazione a
livello di pixel.

### Artefatti

- pacchetto originale:
  `results/anomaly_dino/E05_v1_anomalydino_16shot/E05_v1_anomalydino_16shot_results.zip`;
- export verificato, mappe grezze, tabelle e figure:
  `results/anomaly_dino/E05_v1_anomalydino_16shot/export/`;
- copia NPZ senza pickle:
  `results/anomaly_dino/E05_v1_anomalydino_16shot/export/raw_anomaly_maps_safe.npz`;
- notebook Colab eseguito: `notebooks/E05_AnomalyDINO_v1.ipynb`;
- SHA256 notebook:
  `2d20bdc146c7434ee64faeb5de812bb4773cd360c379c2d8453ce5082c67c15a`;
- SHA256 pacchetto:
  `c560204ae8c0a93e55e57ab30e28e1ef5b0eb6947e7ae48265ad2ebc8f3eca2a`;
- SHA256 NPZ portabile:
  `bfae8ec808bedc7dd52e4862fdbed9a6b8269c66ae42fb26f3862c95148dcc98`.

## E06 — AnomalyVFM DINOv2 zero-shot

- **Data:** 2026-07-05
- **Stato:** completato e archiviato
- **Metodo:** AnomalyVFM DINOv2, vero zero-shot sul dominio ferroviario
- **Codice AnomalyVFM:** commit
  `4da96b493a0b12e4e380fa5bf6573c126342855c`
- **Checkpoint:** `MaticFuc/anomalyvfm_dinov2`, revisione Hugging Face
  `c7698cefc6415ca43c93c3e924e02f6863c57e73`
- **Backbone:** DINOv2 ViT-L/14 Reg, input quadrato 672, mappa 96×96
- **Parametri:** 339.847.171
- **SHA256 pesi AnomalyVFM:**
  `8055083c1883a2753b858bf8656138ddc7016b5e3430ebfdb8b977a550fae53f`
- **SHA256 backbone DINOv2:**
  `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`
- **Hardware:** Kaggle, Tesla T4 `cuda:0` di due GPU disponibili
- **Fine-tuning, riferimenti normali e calibrazione:** no

### Split e protocollo

Il test riusa gli stessi 20 frame normali indipendenti e i 500 frame anomali
San Donato di E05. AnomalyVFM non usa i 16 riferimenti né i 20 frame di
calibrazione di E05. La decisione di frame usa esclusivamente la soglia
nativa `0,5` del modello.

### Classificazione di frame

| TN | FP | FN | TP | Precision | Recall | F1 | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0 | 0 | 500 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |

Gli score normali sono compresi tra `0,432` e `0,471`, mentre quelli anomali
tra `0,602` e `0,727`. I 520 frame hanno richiesto `702,48 s`, pari a circa
`0,74 frame/s`.

### Localizzazione

La soglia nativa spaziale `0,5` produce zero maschere non vuote. Il pointing
game ottiene:

- `7/500 = 0,014` sulle sole box `persona`;
- `65/500 = 0,130` su tutte le annotazioni visibili.

Per rendere comunque confrontabili le mappe con E05 è stato applicato anche
il top 1% adattivo per frame, con area minima 200 pixel sulla mappa
ridimensionata:

| Post-processing | IoU | Precision | Recall | F1 | Frame con almeno un match |
|---|---:|---:|---:|---:|---:|
| soglia nativa 0,5 | 0,3 | 0,000 | 0,000 | 0,000 | 0/500 |
| soglia nativa 0,5 | 0,5 | 0,000 | 0,000 | 0,000 | 0/500 |
| top 1% adattivo | 0,3 | 0,281 | 0,156 | 0,201 | 466/500 |
| top 1% adattivo | 0,5 | 0,100 | 0,055 | 0,071 | 202/500 |

### Confronto E05–E06 e interpretazione

AnomalyVFM ottiene una separazione degli score ancora più netta di
AnomalyDINO, ma localizza molto peggio. Il pointing game passa da `0,956` di
E05 a `0,014`; anche il top 1% adattivo scende da F1 `0,365` a `0,201` a IoU
0,3 e da `0,143` a `0,071` a IoU 0,5. Le mappe qualitative si concentrano
spesso su massicciata, bordi, vegetazione e oggetti caldi anziché sulle
persone.

Il risultato dimostra che uno score di frame perfetto non basta a validare
un metodo di anomaly localization. Normali e anomalie provengono inoltre da
scene differenti e i 500 frame sono temporalmente correlati: AUROC e AP non
possono essere attribuite esclusivamente alla presenza delle anomalie.

### Artefatti

- pacchetto originale:
  `results/anomaly_vfm/E06_v1_anomalyvfm_dinov2_zero_shot/E06_v1_anomalyvfm_dinov2_zero_shot_results.zip`;
- export verificato, NPZ portabile, tabelle e figure:
  `results/anomaly_vfm/E06_v1_anomalyvfm_dinov2_zero_shot/export/`;
- SHA256 pacchetto:
  `9175e8f87e2c2a1db6b5bf74e5150020e70ca8be0c46f9c9d8a56afe7dd3d864`;
- notebook Kaggle eseguito: `notebooks/E06_AnomalyVFM_v1.ipynb`;
- SHA256 notebook:
  `40476a0bae6cf1d9446d9190ab8b20d2655d7de8fc5a3d5a0eb761d0a352034c`.

## E07 — AnomalyVFM sul Thermal Anomaly Detection Dataset

- **Data:** 2026-07-05
- **Stato:** completato e archiviato
- **Dataset:** `neelu1/thermal-anomaly-detection-dataset`, versione 2,
  CC BY 4.0
- **Split:** 300 clip normali di training, 29 clip test
- **Test:** 3.509 frame, dei quali 2.385 normali e 1.124 anomali
- **Ground truth:** etichette temporali binarie; nessuna box o maschera
- **Modello:** `MaticFuc/anomalyvfm_dinov2`, revisione
  `c7698cefc6415ca43c93c3e924e02f6863c57e73`
- **Codice:** commit `4da96b493a0b12e4e380fa5bf6573c126342855c`
- **Hardware:** Kaggle, due Tesla T4, batch 24
- **Fine-tuning:** no
- **Tempo test:** `2.425,86 s`, circa `1,446 frame/s`

### Classificazione di frame

| Protocollo | Soglia | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nativa | 0,500 | 2242 | 143 | 1058 | 66 | 0,316 | 0,059 | 0,099 |
| percentile 95 dei normali | 0,465 | 2117 | 268 | 1000 | 124 | 0,316 | 0,110 | 0,164 |

AUROC è `0,462` e AP `0,309`, leggermente inferiore alla prevalenza positiva
`0,320`. Lo score medio anomalo è `0,382`, contro `0,389` per i normali. La
calibrazione migliora il recall soltanto spostando il punto operativo e non
modifica il ranking.

### Analisi per clip e temporale

La media delle AUROC calcolate separatamente sui 29 clip è `0,444`, con
mediana `0,448`; 13 clip superano 0,5. Lo smoothing a 11 frame porta AUROC
globale a `0,465`, ma F1 di frame alla soglia calibrata scende a `0,138`.

| Metodo | tIoU | Eventi TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| soglia nativa, score grezzo | 0,1 | 4 | 132 | 25 | 0,029 | 0,138 | 0,048 |
| soglia calibrata, score grezzo | 0,3 | 1 | 210 | 28 | 0,005 | 0,034 | 0,008 |
| soglia calibrata, smoothing 11 | 0,3 | 3 | 14 | 26 | 0,176 | 0,103 | 0,130 |

Lo smoothing riduce la frammentazione delle predizioni, ma non recupera la
maggioranza degli eventi.

### Interpretazione

E07 non replica la separazione perfetta di E06 quando normali e anomalie
appartengono agli stessi video. Le mappe evidenziano spesso tetti, luci,
veicoli e sorgenti calde persistenti; alcune persone compaiono anche durante
intervalli normali, confermando che la sola categoria dell'oggetto non
definisce l'anomalia. Il checkpoint industriale zero-shot non generalizza
efficacemente alle anomalie semantiche e comportamentali di questo benchmark.

### Artefatti

- pacchetto finale:
  `results/anomaly_vfm/E07_v1_anomalyvfm_external_zero_shot/E07_v1_anomalyvfm_external_zero_shot_results.zip`;
- export consolidato:
  `results/anomaly_vfm/E07_v1_anomalyvfm_external_zero_shot/export/`;
- checkpoint grezzi base e calibrazione nella stessa cartella;
- SHA256 pacchetto finale:
  `9e50514c0fb6a3aa4f9709dbd0c53d5636bcb6d71e20c26a12fa87332cf3d74f`;
- notebook Kaggle sorgente:
  `notebooks/E07_AnomalyVFM_External_v1.ipynb`;
- SHA256 notebook:
  `52e73eb73d7c06f96af5a9430327654edf7a2f9b0c0a242ebcdf9e2d0d38d34b`;
- nota: il download Kaggle non contiene gli output incorporati; gli output
  eseguiti sono conservati nel pacchetto finale verificato.

## E08 — AnomalyDINO sul Thermal Anomaly Detection Dataset

- **Data:** 2026-07-05
- **Stato:** completato e archiviato
- **Dataset e test:** identici a E07, 29 clip e 3.509 frame, dei quali 2.385
  normali e 1.124 anomali
- **Ground truth:** etichette temporali binarie; nessuna box o maschera
- **Metodo:** AnomalyDINO reference-based, senza fine-tuning
- **Riferimenti:** memorie annidate da 16, 64 e 200 frame centrali di clip
  normali
- **Calibrazione:** percentile 95 di 100 frame centrali appartenenti a clip
  normali disgiunti da quelli di riferimento
- **Seed:** `20260705`
- **Codice AnomalyDINO:** commit
  `b9d1c2648e3a5247437d4d953d907a8f3d994457`
- **Backbone:** DINOv2 ViT-S/14, lato corto 448, griglia 32×42
- **SHA256 pesi:** `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`
- **Hardware:** Kaggle, una Tesla T4 usata e una seconda disponibile
- **Tempo test:** `569,24 s` per tutti i tre assetti, circa `6,164 frame/s`

### Classificazione di frame

| Riferimenti | Soglia | TN | FP | FN | TP | Precision | Recall | F1 | AUROC | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0,445753 | 2306 | 79 | 1044 | 80 | 0,503 | 0,071 | 0,125 | 0,593 | 0,400 |
| 64 | 0,353317 | 2253 | 132 | 1006 | 118 | 0,472 | 0,105 | 0,172 | 0,560 | 0,382 |
| 200 | 0,292679 | 2146 | 239 | 952 | 172 | 0,418 | 0,153 | 0,224 | 0,578 | 0,379 |

Il 16-shot ottiene il migliore ranking globale; il 200-shot ottiene invece
il migliore F1 alla soglia calibrata e la migliore AUROC media per clip
(`0,621`, contro `0,569` del 16-shot). Questa differenza indica che il numero
di riferimenti modifica anche l'offset dello score tra scene e che la sola
AUROC globale non descrive completamente il comportamento sui singoli clip.

### Analisi temporale

Con 200 riferimenti e smoothing centrato a 11 frame vengono prodotti 9
eventi. A tIoU 0,3 risultano 5 TP, 4 FP e 24 FN: precision `0,556`, recall
`0,172` e F1 `0,263`. La continuità temporale riduce nettamente la
frammentazione ma non recupera la maggioranza degli eventi reali.

### Interpretazione provvisoria

AnomalyDINO generalizza meglio di AnomalyVFM E07 a livello di ranking
globale (`0,593` contro `0,462`), ma la separazione resta modesta e il recall
alla soglia calibrata è basso. L'uso di più riferimenti migliora F1 e analisi
intra-clip, senza produrre un miglioramento monotono dell'AUROC globale.
Poiché mancano annotazioni spaziali, le mappe saranno valutate soltanto in
modo qualitativo.

### Analisi qualitativa

Sono stati conservati due veri positivi, due falsi positivi e due falsi
negativi del 200-shot, selezionati su clip distinti, e due confronti diretti
fra mappe 16/64/200-shot. Le risposte più intense coincidono spesso con
veicoli, illuminazione, tetti e altre sorgenti calde. Nei falsi positivi la
salienza termica dello sfondo supera la soglia; nei falsi negativi l'evento
annotato non si distingue sufficientemente dalla memoria normale. Le mappe
sono normalizzate per frame soltanto per la visualizzazione.

### Artefatti

- pacchetto finale verificato:
  `results/anomaly_dino/E08_v1_anomalydino_external_nested_shots/E08_v1_anomalydino_external_nested_shots_results.zip`;
- checkpoint base verificato:
  `results/anomaly_dino/E08_v1_anomalydino_external_nested_shots/E08_base_checkpoint.zip`;
- tabelle e figura quantitativa:
  `results/anomaly_dino/E08_v1_anomalydino_external_nested_shots/export/`;
- SHA256 pacchetto finale:
  `675d1cde8a69ab92989813c9bc0871b0a98ddbbc4202e336e93a197cafb35fb0`;
- SHA256 checkpoint base:
  `8de98304bf0cadcaa0d72317c9b68eefaded2fda0037893b65285008028cd32d`;
- SHA256 checkpoint qualitativo:
  `86be09072a53af1ddb599eea1a943d5d33f64bc4b6e2e54af26fb82a9f0b1a18`;
- notebook Kaggle eseguito, con output incorporati:
  `notebooks/E08_AnomalyDINO_External_v1.ipynb`;
- SHA256 notebook:
  `807fb79884ed932f67ffa90047a23bafcaa89949b2f65f06511840012dbe0378`;
- script riproducibile di post-processing:
  `src/analyze_e08_checkpoint.py`.

## E09 — AnomalyDINO point to SAM2

- **Data:** 2026-07-06
- **Stato:** completato e archiviato
- **Sorgente:** mappe e score congelati di E05 AnomalyDINO 16-shot
- **Test:** 500 frame consecutivi San Donato, frame sorgente 2125–2624
- **Prompt:** un punto positivo automatico, massimo della mappa AnomalyDINO
- **Baseline spaziale:** componente top 1% contenente lo stesso punto
- **Modello:** SAM 2.1 Small, image predictor indipendente per frame
- **Selezione multimask:** maggiore IoU predetta da SAM2
- **Ground truth:** box `persona` CVAT v2 revisionate
- **Fine-tuning e prompt manuali:** no
- **Controllo negativo:** 20 frame normali indipendenti di E05
- **Hardware:** Kaggle, Tesla T4

### Raffinamento del candidato

| Metodo | IoU media | IoU mediana | Hit rate IoU≥0,3 | Hit rate IoU≥0,5 |
|---|---:|---:|---:|---:|
| componente AnomalyDINO | 0,501 | 0,506 | 0,928 | 0,534 |
| SAM2 da punto automatico | 0,717 | 0,757 | 0,976 | 0,934 |

SAM2 migliora 457 frame su 500, ne lascia uno invariato e ne peggiora 42.
Il guadagno medio di IoU è `+0,215`. Nei 478 frame in cui il punto cade già
all'interno di una persona, SAM2 raggiunge hit rate `0,946` a IoU 0,5.

### Lettura detection-style e limite del singolo punto

Con una sola predizione per frame, a fronte di 3.732 box persona:

| Metodo | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| componente AnomalyDINO | 0,3 | 0,928 | 0,124 | 0,219 |
| SAM2 da punto automatico | 0,3 | 0,976 | 0,131 | 0,231 |
| componente AnomalyDINO | 0,5 | 0,534 | 0,072 | 0,126 |
| SAM2 da punto automatico | 0,5 | 0,934 | 0,125 | 0,221 |

Il recall resta basso perché un punto recupera al massimo un oggetto, mentre
in molti frame sono visibili più persone. E09 dimostra quindi il
raffinamento del candidato selezionato, non una soluzione multi-oggetto
completa.

### Controllo negativo e interpretazione

Quattro dei 20 frame normali superano già la soglia E05. SAM2 segmenta una
regione della massicciata in tutti e quattro e non modifica la decisione di
frame: i falsi positivi restano `4/20`. Analogamente, quando il punto massimo
cade sulla regione sbagliata, SAM2 segmenta accuratamente l'oggetto sbagliato.
Il raffinamento migliora dunque la geometria di un candidato valido, ma non
corregge gli errori semantici o di detection del modulo a monte.

### Runtime e artefatti

- 504 maschere in `98,02 s`, circa `5,142 maschere/s`;
- picco GPU `0,61 GB`;
- commit SAM2:
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- SHA256 checkpoint SAM2:
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`;
- pacchetto finale:
  `results/sam_refinement/E09_v1_anomalydino_point_sam2/E09_v1_anomalydino_point_sam2_results.zip`;
- SHA256 pacchetto finale:
  `93fc9d4aec71d022b0bad76dfbce4a16be8d0bacb55ff174a9011f2e375bed0c`;
- checkpoint base SHA256:
  `f4a4f9f1d3f642381ef2669b747f0e28d896ba95004078cf26691d7f85dcb053`;
- notebook Kaggle eseguito:
  `notebooks/E09_AnomalyDINO_SAM2_v1.ipynb`;
- SHA256 notebook:
  `476a8f62168c6eb614a9727972d95e2767bcc70e3daa20d2aaf8a3fd5cb823d9`;
- script di post-processing: `src/analyze_e09_checkpoint.py`.

## E10 — AnomalyDINO multi-punto to SAM2

- **Data:** 2026-07-06
- **Stato:** completato, verificato e archiviato
- **Sorgente:** stessi input congelati di E09/E05
- **Domanda:** più massimi AnomalyDINO aumentano la copertura delle 3.732 box
  persona e delle 1.000 box `incerto_sui_binari` senza rendere inutilizzabile
  la precisione?
- **Configurazioni annidate:** `K=1`, `K=3`, `K=5`
- **Primo punto:** identico al massimo congelato E09
- **Punti successivi:** massimi della mappa ridimensionata, distanza minima
  `64 px`
- **Deduplicazione:** box IoU `0,7`, preservando l'ordine dei punti
- **Matching:** greedy uno-a-uno a IoU `0,3` e `0,5`
- **Fine-tuning e prompt manuali:** no
- **Controllo negativo:** stessi 20 normali E09, con gate di frame E05
- **Scope separati:** `persona`, `incerto_all`, `incerto_sui_binari`,
  `all_sui_binari`, `all_visible`

Il valore di `K` non è stato selezionato guardando il test: sono riportati
tutti e tre i punti del compromesso precision–recall. K=1 replica esattamente
E09. Sulle persone, K=3 offre il migliore F1; K=5 aumenta ancora il recall ma
riduce la precisione.

| Scope persona, IoU 0,3 | Precision | Recall | F1 |
|---|---:|---:|---:|
| K=1 | 0,976 | 0,131 | 0,231 |
| K=3 | 0,844 | 0,298 | 0,440 |
| K=5 | 0,583 | 0,318 | 0,412 |

Per `incerto_sui_binari`, K=1, K=3 e K=5 ottengono tutti `0` TP, precision
`0`, recall `0` e F1 `0` sia a IoU 0,3 sia a IoU 0,5. Le configurazioni
producono rispettivamente 500, 1.317 e 2.038 predizioni dopo deduplicazione,
senza recuperare nessuna delle 1.000 box target. L'aumento dei candidati
recupera quindi altre persone e regioni di sfondo, non gli oggetti incerti.

Il controllo negativo conserva quattro falsi positivi di frame su 20; le
proposte spaziali dopo deduplicazione sono 4, 8 e 13 per K=1,3,5.
L'inferenza di 2.520 maschere richiede `121,06 s`, pari a `20,816`
maschere/s, con picco GPU `0,61 GB`.

Durante la prima generazione della figura qualitativa il kernel Kaggle ha
esaurito la RAM perché conservava le maschere booleane non compresse. Il
checkpoint era già stato salvato; la figura è stata rigenerata dalle box e
il notebook riproducibile è stato corretto eliminando le maschere dalla RAM
dopo il packbits.

Artefatti:

- protocollo:
  `docs/protocollo_e10_anomalydino_multipoint_sam2.md`;
- notebook Kaggle:
  `notebooks/E10_AnomalyDINO_MultiPoint_SAM2_v1.ipynb`;
- notebook eseguito:
  `notebooks/E10_AnomalyDINO_MultiPoint_SAM2_v1_executed.ipynb`;
- risultati:
  `results/sam_refinement/E10_v1_anomalydino_multipoint_sam2/`;
- SHA256 pacchetto finale:
  `d8faa2b82ed54f4e65a6220b0f8aac3b5e4e256f437b8658f461569888663519`;
- SHA256 notebook eseguito:
  `0027817f7fd96311babc51e011ba325805f1f73517b49fbd313657d22ab5f0c5`;
- SHA256 notebook riproducibile memory-safe:
  `829a5ee703dc90fae8fb8442ee295b7a605c8e73c1fb35e5d67e759b05695e3a`.
