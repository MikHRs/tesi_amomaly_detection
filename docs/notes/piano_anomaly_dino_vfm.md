# Piano esperimenti AnomalyDINO / AnomalyVFM

## Obiettivo

Valutare preliminarmente se feature di foundation model visivi consentono di
rilevare e localizzare anomalie in immagini termiche, e se una regione
candidata può essere raffinata con SAM/SAM2.

Le domande sperimentali sono:

1. il metodo assegna score maggiori ai frame anomali rispetto ai normali?
2. i massimi della anomaly map cadono sulle box CVAT?
3. quali pattern generano falsi positivi o falsi negativi?
4. SAM migliora la forma della regione senza recuperare candidati mancati?

## Dati normali

Sorgente principale:

```text
data/raw/ir_2025-01-31_04-15-15.mp4
```

Estrarre circa 200 frame distanziati:

```bash
python src/extract_normal_frames.py \
  --video data/raw/ir_2025-01-31_04-15-15.mp4 \
  --out data/normal_train_frames \
  --num-frames 200
```

Prima dell'uso:

- verificare che non contengano anomalie evidenti;
- conservare `frames.csv`;
- non scegliere frame guardando i risultati del modello;
- riservare un intervallo normale indipendente per stimare i falsi positivi.

Questi frame sono riferimenti per AnomalyDINO. Non vengono usati per
adattare AnomalyVFM.

## Dati test

Le sorgenti anomale sono i due video con ground truth CVAT:

```text
data/raw/ir_2025-01-31_05-09-12.mp4
data/raw/test_video_san_donato.mp4
```

Estrarre frame con timestamp o indici congelati prima della valutazione:

```bash
python src/extract_test_anomaly_frames.py \
  --video data/raw/test_video_san_donato.mp4 \
  --out data/test_anomaly_frames \
  --times 75,85,95,110,125
```

Gli esempi numerici del comando sono dimostrativi: i timestamp definitivi
devono essere ricavati dalle annotazioni revisionate. Servono inoltre frame
test normali non inclusi nella memoria di riferimento.

## Metodo

### Percorso A — AnomalyDINO

AnomalyDINO usa patch feature DINOv2 e una memoria di normalità. È
training-free, ma con riferimenti termici del dominio non va descritto come
zero-shot puro.

Protocollo:

1. costruire la memoria con i frame normali;
2. fissare backbone, layer, risoluzione, numero di patch e preprocessing;
3. scegliere la soglia solo su normalità di validation;
4. eseguire una volta sul test congelato;
5. salvare score e mappe grezze prima di qualsiasi filtro.

### Percorso B — AnomalyVFM

AnomalyVFM usa checkpoint già adattati su dati ausiliari e produce
predizioni zero-shot senza immagini normali del dominio. Il confronto deve
registrare il backbone scelto e la versione dei pesi.

Protocollo:

1. installare il repository ufficiale o la versione verificata di Anomalib
   in un ambiente Colab separato;
2. caricare soltanto i frame test;
3. non effettuare fine-tuning sui video ferroviari;
4. salvare score, mappe, maschere e configurazione.

## Output attesi

Per ogni frame:

- anomaly score;
- anomaly map non sogliata;
- maschera ottenuta con soglia dichiarata;
- overlay qualitativo;
- metadati con metodo, versione, backbone, checkpoint e soglia.

Struttura suggerita:

```text
results/anomaly_dino/<experiment_id>/
results/anomaly_vfm/<experiment_id>/
```

Non sovrascrivere output di esperimenti differenti.

## Confronto con CVAT

La label è class-agnostic: tutte le box attive sono `anomalia`. Valutazioni
possibili:

- score di frame su test normale e anomalo;
- AUROC e average precision solo se il test contiene entrambe le classi;
- recall, precision e F1 a una soglia scelta senza usare il test;
- pointing game tra massimo della heatmap e box CVAT;
- conversione map→box e matching a IoU 0,3 e 0,5;
- analisi separata di oggetti piccoli, lontani o parzialmente visibili.

Le metriche pixel-level richiedono maschere ground truth e non possono essere
dedotte dalle sole box.

## Raffinamento con SAM

Per un sottoinsieme congelato di frame:

```text
anomaly map
  → soglia
  → componenti connesse
  → box/punto candidato
  → SAM/SAM2
  → maschera raffinata
```

Conversione iniziale:

```bash
python src/anomaly_output_to_box.py \
  --mask results/anomaly_vfm/frame_001_mask.png \
  --out results/anomaly_vfm/frame_001_box.csv \
  --threshold 0.5 \
  --min-area 50
```

Confrontare output grezzo e raffinato sugli stessi frame. SAM può migliorare
il contorno di una regione presente, ma non risolve un falso negativo se la
anomaly map non propone alcun candidato.

## Metriche/valutazione qualitativa

Compilare il template:

```bash
python src/build_experiment_table.py \
  --out results/tables/experiment_summary.csv
```

Valori ammessi per `localization_quality`:

```text
good, partial, bad, not_detected, uncertain
```

Mostrare esempi di veri positivi, falsi positivi e falsi negativi. Dare
particolare evidenza ai falsi negativi, senza eliminare dalle tabelle i casi
sfavorevoli.

## Limiti

- I metodi sono nati soprattutto su immagini industriali RGB, non su scene
  ferroviarie termiche.
- Il domain shift può produrre score elevati sullo sfondo, sulla vegetazione
  o su regioni calde.
- Frame vicini dello stesso video sono fortemente correlati.
- Una soglia minima di area penalizza oggetti piccoli o lontani.
- Le box CVAT descrivono localizzazione grossolana, non segmentazione
  pixel-perfect.
- Il confronto rimane una valutazione preliminare e non dimostra capacità di
  impiego operativo.
