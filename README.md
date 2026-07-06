# Localizzazione di anomalie in immagini e video termici

Repository leggera e riproducibile per una tesi triennale in Ingegneria
Informatica. Il progetto studia anomaly detection/localization, foundation
model visuali e segmentazione prompt-based nel dominio termico. I video
ferroviari costituiscono il caso di studio principale, ma il titolo e il
protocollo non sono limitati a un solo scenario applicativo.

La label principale è `anomalia`. Nell'export CVAT v2 revisionato, l'attributo
`tipo` contiene soltanto `persona` e `incerto`; parole come `box` e `pallet`
sono state usate nei prompt esplorativi E03, non come classi ground truth.
SAM/SAM2 riceve un candidato tramite punto, box o maschera e lo segmenta o
propaga, ma non decide da solo che cosa sia anomalo.

## Vincolo scientifico del progetto

Il compito finale è **anomaly detection/localization**, non object detection.
Grounding DINO può proporre regioni e SAM2 può segmentarle e seguirle, ma il
riconoscimento di una persona, una scatola o un pallet non costituisce da solo
una decisione di anomalia. L'output finale deve esprimere quanto una regione o
un evento si discosti dalla normalità della scena, considerando anche
posizione e comportamento temporale. I risultati di object detection sono
quindi soltanto risultati intermedi o baseline ausiliarie.

## Pipeline aggiornata dopo incontro con il relatore

La pipeline finale considera tre componenti:

1. CVAT per annotazioni manuali preliminari;
2. SAM/SAM2 per segmentazione prompt-based, tracking e raffinamento;
3. AnomalyDINO/AnomalyVFM per la localizzazione automatica di anomalie.

Il flusso sperimentale è:

```text
video termici
  → frame normali
  → AnomalyDINO / AnomalyVFM
  → anomaly score + anomaly map/mask
  → box o punto candidato
  → SAM/SAM2
  → confronto con annotazioni CVAT
```

I due anomaly detector non hanno lo stesso protocollo:

- AnomalyDINO è training-free ma usa immagini normali di riferimento del
  dominio; con circa 200 frame ferroviari normali è un esperimento
  reference-based, non zero-shot puro;
- AnomalyVFM usa checkpoint adattati su dati ausiliari e viene valutato
  zero-shot, senza adattamento sui frame ferroviari.

E05 ha eseguito AnomalyDINO 16-shot: F1 di frame `0,996` e pointing game
`0,956` sulla clip San Donato. Il risultato è preliminare perché normali e
anomalie provengono da scene differenti; la separazione perfetta degli score
può includere domain shift dello sfondo.

E06 ha valutato AnomalyVFM DINOv2 in vero zero-shot sul medesimo test
congelato. La soglia nativa `0,5` separa tutti i 20 frame normali dai 500
anomali, ma la localizzazione fallisce: il pointing game sulle persone è
soltanto `0,014` e le maschere native sono vuote in tutti i frame. Questo
contrasto mostra perché classificazione di frame e localizzazione
dell'anomalia devono essere valutate separatamente.

E07 ha ripetuto AnomalyVFM sul Thermal Anomaly Detection Dataset, nel quale
normali e anomalie appartengono agli stessi 29 clip. AUROC scende a `0,462`
e F1 con soglia nativa a `0,099`; una soglia calibrata su 300 clip normali
porta F1 soltanto a `0,164`. Lo smoothing temporale non corregge
l'ordinamento degli score. Il confronto E06–E07 mostra quindi che la
separazione perfetta tra due video non costituiva evidenza sufficiente di
generalizzazione.

E08 ha valutato AnomalyDINO sullo stesso benchmark con memorie annidate da
16, 64 e 200 riferimenti e 100 clip normali disgiunti per la calibrazione.
Il migliore ranking globale è AUROC `0,593` con 16 riferimenti; il migliore
F1 calibrato è `0,224` con 200 riferimenti. AnomalyDINO migliora rispetto a
E07, ma il recall rimane basso e il vantaggio non è monotono su tutte le
metriche.

E09 ha collegato i due moduli sul caso ferroviario: il massimo della mappa
AnomalyDINO viene usato come punto automatico per SAM2, senza prompt manuali.
L'IoU media della box passa da `0,501` a `0,717` e l'hit rate a IoU 0,5 da
`0,534` a `0,934`. SAM2 migliora quindi la forma del candidato selezionato,
ma non elimina i falsi positivi di frame e non corregge un punto posto sulla
regione sbagliata.

E10 ha misurato il limite multi-oggetto di E09 con proposte annidate
`K=1,3,5`, massimi AnomalyDINO separati di 64 px e deduplicazione delle box
SAM2 a IoU `0,7`. Sulle persone, K=3 porta il recall da `0,131` a `0,298` e
F1 da `0,231` a `0,440` a IoU 0,3; K=5 raggiunge recall `0,318` ma perde
precisione. Sugli oggetti `incerto_sui_binari`, tutte le configurazioni
ottengono `0` true positive sia a IoU 0,3 sia a 0,5. Il risultato conferma
che l'aumento dei candidati recupera altre persone, ma non gli oggetti
incerti annotati sui binari.

## Cosa contiene la pipeline

- inventario tecnico dei video;
- estrazione di frame e creazione di clip;
- ispezione e conversione di export XML CVAT;
- controllo visivo delle bounding box;
- IoU, matching greedy uno-a-uno, precision, recall e F1;
- configurazione di prompt diretti, dello sfondo e combinati;
- estrazione di frame normali distanziati e frame test selezionati;
- conversione di anomaly mask in box candidate;
- template per l'analisi di TP, FP e FN;
- documentazione per annotazione, tesi ed esperimenti;
- stub SAM2 e AnomalyVFM sicuri, senza download automatici.

## Struttura

```text
tesi_anomaly_detection/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── prompts_zero_shot.yaml
│   └── experiment_config.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── work/
│   ├── frames/
│   ├── clips/
│   ├── exports_cvat/
│   ├── normal_train_frames/
│   ├── test_anomaly_frames/
│   ├── cvat_annotations/
│   ├── sam_outputs/
│   └── anomaly_outputs/
├── docs/
│   ├── protocollo_annotazione_v0.md
│   ├── struttura_tesi.md
│   ├── bibliografia_iniziale.md
│   ├── linee_guida_unifi.md
│   ├── domande_professore.md
│   ├── appunti_sam_sam2.md
│   ├── diario_tesi.md
│   └── notes/
├── templates/
│   └── unifi/
│       ├── tesi_latex/
│       └── presentazione_16_9.pptx
├── src/
│   ├── __init__.py
│   ├── video_inventory.py
│   ├── extract_frames.py
│   ├── extract_normal_frames.py
│   ├── extract_test_anomaly_frames.py
│   ├── make_clip.py
│   ├── inspect_cvat_export.py
│   ├── cvat_to_boxes.py
│   ├── visualize_boxes.py
│   ├── anomaly_output_to_box.py
│   ├── build_experiment_table.py
│   ├── metrics.py
│   ├── evaluate_predictions.py
│   ├── prompt_strategy.py
│   ├── sam2_notes_or_stub.py
│   └── anomaly_vfm_notes_or_stub.py
├── notebooks/
│   └── README.md
├── tests/
└── results/
    ├── inventory/
    ├── frames_preview/
    ├── visualizations/
    ├── metrics/
    ├── qualitative/
    ├── anomaly_vfm/
    ├── anomaly_dino/
    ├── sam_refinement/
    └── tables/
```

I contenuti di `data/` (tranne il README) e `results/` sono ignorati da Git.
Non inserire video, export sensibili, checkpoint o risultati pesanti nei
commit.

## Setup

Da eseguire dalla radice del repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Gli ambienti virtuali non sono portabili tra percorsi diversi. Se la cartella
del progetto è stata spostata, rinominare o eliminare intenzionalmente la
vecchia `.venv` e crearne una nuova invece di correggere i collegamenti a
mano.

Verifica rapida:

```bash
python -m unittest discover -v
python src/metrics.py
python src/prompt_strategy.py --config configs/prompts_zero_shot.yaml
```

## Flusso consigliato

### 1. Inserire i video

Copiare i video ricevuti, senza rinominarli se non concordato, in
`data/raw/`. Conservare altrove una copia originale in sola lettura. Prima di
usare CVAT Online, Colab, RunPod o altri cloud chiedere l'autorizzazione del
relatore.

### 2. Creare l'inventario

```bash
python src/video_inventory.py \
  --input data/raw \
  --output results/inventory/inventario_video.csv
```

Il CSV contiene nome, percorso, stato di lettura, durata, fps, numero di
frame, risoluzione, codec, note ed eventuale errore.

### 3. Creare una clip breve per il pilot

```bash
python src/make_clip.py \
  --video data/raw/video.mp4 \
  --start 10 \
  --end 25 \
  --out data/clips/video_clip_10_25.mp4
```

L'intervallo finale è escluso. OpenCV mantiene gli fps quando possibile, ma
non conserva l'audio. Per la tesi l'audio non è necessario.

### 4. Estrarre frame

```bash
python src/extract_frames.py \
  --video data/raw/video.mp4 \
  --out data/frames/video1 \
  --fps 1
```

Solo una porzione:

```bash
python src/extract_frames.py \
  --video data/raw/video.mp4 \
  --out data/frames/video1_10_30 \
  --fps 1 \
  --start 10 \
  --end 30
```

Ogni cartella contiene immagini come
`frame_000001_t000.00.jpg` e `frames.csv`. `frame_index` è l'indice
zero-based del video sorgente; il timestamp è in secondi.

### 4.1 Estrarre i riferimenti normali

Il video normale indicato dal relatore è il terzo video non annotato. Va
comunque controllato prima di assumere che sia interamente normale:

```bash
python src/extract_normal_frames.py \
  --video data/raw/ir_2025-01-31_04-15-15.mp4 \
  --out data/normal_train_frames \
  --num-frames 200
```

Lo script salva `normal_0001.jpg`, …, `normal_0200.jpg` e `frames.csv`. I
frame sono distribuiti sull'intera durata e non sono consecutivi.

### 4.2 Estrarre frame test selezionati

Per timestamp:

```bash
python src/extract_test_anomaly_frames.py \
  --video data/raw/test_video_san_donato.mp4 \
  --out data/test_anomaly_frames \
  --times 75,85,95,110,125
```

Oppure per indice:

```bash
python src/extract_test_anomaly_frames.py \
  --video data/raw/test_video_san_donato.mp4 \
  --out data/test_anomaly_frames \
  --frames 1875,2125,2375
```

I valori sono esempi e non una selezione sperimentale definitiva.

### 5. Annotare con CVAT

Per il pilot:

1. caricare una clip breve o una cartella di frame;
2. creare la label `anomalia` e gli attributi in
   `configs/experiment_config.yaml`;
3. usare box track, `occluded` e `outside`;
4. esportare in formato XML nativo CVAT;
5. estrarre `annotations.xml` dallo ZIP, se necessario, in
   `data/exports_cvat/`.

Se si annota una clip, estrarre per il controllo visivo i frame dalla stessa
clip: CVAT numera i frame della clip a partire da zero.

### 6. Ispezionare e convertire l'export

```bash
python src/inspect_cvat_export.py \
  --xml data/exports_cvat/annotations.xml

python src/cvat_to_boxes.py \
  --xml data/exports_cvat/annotations.xml \
  --out results/annotations_boxes.csv
```

Il secondo comando produce:

```text
video_or_image,frame,track_id,label,xtl,ytl,xbr,ybr,outside,occluded
```

Le righe `outside=1` restano nel CSV per documentare la track, ma vengono
escluse dalla visualizzazione e dalla valutazione.

### 7. Controllare le box

```bash
python src/visualize_boxes.py \
  --frames data/frames/video1 \
  --boxes results/annotations_boxes.csv \
  --out results/visualizations
```

Controllare almeno primo/ultimo frame di ogni track, transizioni `outside`,
occlusioni e alcuni frame casuali.

### 8. Preparare le predizioni

Il CSV minimo richiesto è:

```text
frame,label,score,xtl,ytl,xbr,ybr,prompt,method
0,anomalia,0.82,10,20,80,120,person,metodo_zero_shot
```

Per evitare collisioni tra indici di frame, usare un CSV per video/clip o
garantire una numerazione globale coerente. Le predizioni non vengono create
da questo repository: devono provenire da un metodo realmente eseguito.

### 9. Calcolare le metriche

```bash
python src/evaluate_predictions.py \
  --gt results/annotations_boxes.csv \
  --pred results/predictions_boxes.csv \
  --out results/metrics/evaluation.csv \
  --iou 0.5
```

Per una seconda soglia:

```bash
python src/evaluate_predictions.py \
  --gt results/annotations_boxes.csv \
  --pred results/predictions_boxes.csv \
  --out results/metrics/evaluation_iou_03.csv \
  --iou 0.3
```

La valutazione predefinita è class-agnostic, coerente con `anomalia`. Lo
script salva una riga globale e aggregazioni per metodo, prompt e coppia
metodo-prompt. `--match-label` è disponibile soltanto per esperimenti in cui
le label di predizione e ground truth hanno davvero lo stesso significato.

Attenzione: la riga globale mescola tutte le predizioni del file. Se il CSV
contiene esperimenti alternativi, confrontare principalmente le righe
`method_prompt`, non interpretare l'unione come un singolo detector.

### 10. Conservare i risultati per la tesi

- CSV/JSON tecnici in `results/inventory/` e `results/metrics/`;
- esempi TP, FP e FN in `results/qualitative/`;
- immagini con box in `results/visualizations/`;
- configurazione, prompt, checkpoint, soglie, hardware e data nel diario.

Non modificare a mano un risultato senza conservarne origine e comando.

## Strategie zero-shot

Mostrare i prompt configurati:

```bash
python src/prompt_strategy.py \
  --config configs/prompts_zero_shot.yaml
```

Le tre famiglie sono:

- prompt diretti sul candidato anomalo;
- prompt sullo sfondo normale, da cui ricavare residui candidati;
- prompt combinati.

SAM e SAM2 non accettano nativamente questi prompt testuali. Una pipeline
possibile è detector open-vocabulary → box → SAM/SAM2 → maschera/track. La
strategia sullo sfondo richiede inoltre una regola esplicita per creare blob
residui; non è implementata automaticamente.

Controllare lo stub:

```bash
python src/sam2_notes_or_stub.py
```

## Preparazione del pilot one-class E04

Il dataset derivato per la prima baseline di anomaly detection si rigenera con:

```bash
python src/build_anomaly_dataset.py \
  --inventory results/inventory/inventario_video.csv \
  --annotations-dir results/annotations \
  --out data/processed/anomaly_pilot_v3 \
  --sample-fps 2 \
  --margin-seconds 2 \
  --train-fraction 0.8 \
  --jpeg-quality 90
```

Il training contiene esclusivamente frame normali. Le box CVAT dei frame
anomali vengono esportate in `ground_truth_boxes.csv` solo per la valutazione.

E04 resta un pilot storico: usa frame normali ricavati dagli stessi due video
annotati e il notebook corrente combina validation normale e test anomalo
nelle metriche. Non va presentato come protocollo finale. La nuova
valutazione usa il terzo video come sorgente normale separata e deve
riservare normali indipendenti per stimare i falsi positivi.

## Preparazione AnomalyDINO / AnomalyVFM

La nota eseguibile chiarisce input e output senza scaricare modelli:

```bash
python src/anomaly_vfm_notes_or_stub.py
```

Dopo un'inferenza realmente eseguita, una maschera può essere convertita in
box candidate:

```bash
python src/anomaly_output_to_box.py \
  --mask results/anomaly_vfm/frame_001_mask.png \
  --out results/anomaly_vfm/frame_001_box.csv \
  --threshold 0.5 \
  --min-area 50
```

Per la revisione qualitativa:

```bash
python src/build_experiment_table.py \
  --out results/tables/experiment_summary.csv
```

PyTorch, SAM2 e i checkpoint non fanno parte dei requisiti principali. Per
una prova futura usare un ambiente separato e le istruzioni del
[repository ufficiale SAM2](https://github.com/facebookresearch/sam2).

## Paper e dataset forniti

Il paper di Li et al.,
[*Segmenting Objects in Day and Night: Edge-Conditioned CNN for Thermal
Image Semantic Segmentation*](https://doi.org/10.1109/TNNLS.2020.3009373),
è utile per motivare segmentazione termica, informazione dei bordi e dataset
SODA. Non è però un metodo zero-shot di anomaly detection ferroviaria.

La survey di Kütük e Algan,
[*Semantic Segmentation for Thermal Images: A Comparative
Survey*](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Kutuk_Semantic_Segmentation_for_Thermal_Images_A_Comparative_Survey_CVPRW_2022_paper.html),
supporta la discussione su thermal crossover, rumore, bassa risoluzione e
bordi ambigui. Non è un lavoro di anomaly detection.

Il
[*Thermal Anomaly Detection Dataset*](https://www.kaggle.com/datasets/neelu1/thermal-anomaly-detection-dataset)
è un benchmark esterno di videosorveglianza termica open-set, con sequenze
normali di training e sequenze anomale di test derivate da *Seasons in
Drift*. Le annotazioni sono temporali e non equivalgono a box o maschere.
Non è ferroviario, non viene scaricato automaticamente e va usato, se serve,
soltanto come controllo esterno dichiarando il domain gap.

La lettura critica di tutti i riferimenti inviati dal relatore è in
`docs/notes/fonti_relatore.md`.

## Cosa mostrare al professore

1. `docs/ricevimento_2026-07-07/00_Relazione_CHIARA.pdf`, sintesi di otto
   pagine che distingue subito persone e oggetti `incerto`;
2. `docs/ricevimento_2026-07-07/LEGGIMI.md`, copione di 10–15 minuti;
3. la pagina 2, che mostra esplicitamente le box `persona` e `incerto`;
4. la pagina 5, che documenta `478` hit persona ma `0` hit incerto;
5. la pagina 6, con il protocollo E10 corretto;
6. soltanto su richiesta, i due video E01/E02 e la relazione tecnica lunga.

`docs/relazione_tecnica_lavoro_svolto.pdf` resta uno storico precedente. Il
documento principale per il ricevimento è
`docs/Relazione_CHIARA_Michele_Rais_2026-07-07.pdf`; la relazione di 24
pagine resta un allegato tecnico.

## Cosa NON fa ancora la pipeline

- E04, il primo pilot che produce un anomaly score rispetto alla normalità,
  è preparato ma non ancora eseguito;
- E06 AnomalyVFM separa perfettamente i due video a livello di score, ma il
  test non consente di separare l'effetto delle anomalie dal cambio globale
  di scena;
- le mappe E06 non forniscono candidati abbastanza affidabili per descrivere
  un raffinamento SAM come pipeline risolutiva;
- E05 AnomalyDINO usa normali e anomalie da scene differenti e non consente
  di attribuire l'AUROC perfetta esclusivamente agli oggetti anomali;
- gli script CLI leggeri non eseguono SAM2 o Grounding DINO: i pilot pesanti
  sono documentati nei notebook Colab separati;
- non scarica modelli o dataset;
- non esegue fine-tuning;
- non implementa ancora la sottrazione dello sfondo o i blob residui;
- E02-v2 valuta soltanto le tre track `persona` abbinate; non è una misura di
  anomaly detection generale e le track `incerto` restano escluse;
- E05 ed E09 localizzano soprattutto persone: l'audit congelato trova
  `478/500` massimi sulle persone e `0/500` sugli oggetti `incerto`; anche le
  proposte E05 hanno `0` match incerto a IoU 0,3;
- non sostituisce la revisione umana delle annotazioni.
