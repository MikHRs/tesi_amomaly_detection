# Contesto rapido per Codex

Questo progetto è la repository della tesi di Michele Rais su anomaly
detection/localization in immagini e video termici, con caso di studio
ferroviario e benchmark termico esterno.

## Obiettivo

Studiare se foundation model visuali e segmentazione prompt-based possono
aiutare a localizzare anomalie in video termici. Il compito finale non è
object detection pura: è anomaly detection/localization.

## Struttura della repository

- `configs/`: configurazioni e prompt.
- `data/`: dati locali, video, frame, export CVAT e dataset processati.
  Molti contenuti sono ignorati da Git perché pesanti o sensibili.
- `src/`: script Python riutilizzabili per inventario video, conversione
  CVAT, estrazione frame, costruzione dataset, metriche e visualizzazioni.
- `notebooks/`: notebook sperimentali E01-E10, eseguiti soprattutto su
  Colab/Kaggle.
- `results/`: output degli esperimenti, CSV, figure, video, zip e metadata.
  È ignorata da Git e va ripristinata da archivio separato.
- `docs/`: protocolli, relazioni, appunti e materiale per ricevimento.
- `tests/`: test automatici.
- `templates/`: template UniFi/LaTeX/PPT.

## Dati e annotazioni

La ground truth principale su San Donato proviene da CVAT. Le annotazioni non
sono maschere pixel-level: sono bounding box con attributi.

File importante:

```text
results/annotations/test_video_san_donato_v2_reviewed_boxes.csv
```

Colonne principali:

```text
frame, track_id, label, xtl, ytl, xbr, ybr, tipo, visibilita, posizione
```

La label è `anomalia`; l'attributo `tipo` distingue `persona` e `incerto`.

## Esperimenti principali

- E01: SAM2 con box manuale su una persona.
- E02: Grounding DINO periodico + SAM2 su persone.
- E03: sensibilità dei prompt Grounding DINO.
- E05: AnomalyDINO 16-shot su clip ferroviaria.
- E06: AnomalyVFM zero-shot su clip ferroviaria.
- E07: AnomalyVFM su Thermal Anomaly Detection Dataset.
- E08: AnomalyDINO su Thermal Anomaly Detection Dataset.
- E09: massimo AnomalyDINO come punto automatico per SAM2.
- E10: multi-punto AnomalyDINO → SAM2 con K=1,3,5.

## Risultati chiave

E09 mostra che SAM2 migliora la forma del candidato se il punto automatico è
corretto:

- IoU media candidato: circa `0,501` → `0,717`;
- hit rate IoU≥0,5: circa `0,534` → `0,934`.

E10 testa più candidati:

- K=1,3,5 massimi AnomalyDINO separati di 64 px;
- deduplicazione box SAM2 a IoU `0,7`;
- metriche a IoU `0,3` e `0,5`;
- 500 frame anomali e 20 frame normali di controllo;
- 2.520 maschere SAM2 generate in circa 121 s.

Risultato E10 sulle persone a IoU 0,3:

- K=1: precision `0,976`, recall `0,131`, F1 `0,231`;
- K=3: precision `0,844`, recall `0,298`, F1 `0,440`;
- K=5: precision `0,583`, recall `0,318`, F1 `0,412`.

Sugli oggetti `incerto_sui_binari`, E10 ottiene `0` true positive per tutte le
configurazioni. Questo è un limite importante: la pipeline funziona meglio
sulle persone che sugli ostacoli piccoli/incerti sui binari.

## Frase scientifica centrale

Il lavoro separa tre problemi:

1. capire se un frame è anomalo;
2. proporre dove guardare;
3. segmentare/localizzare la regione candidata.

SAM2 è forte nel terzo punto, ma dipende dalla qualità del candidato fornito
da AnomalyDINO/AnomalyVFM.

## Come riprendere su un altro PC

Dopo aver clonato la repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Poi ripristinare separatamente `data/` e `results/` dall'archivio creato sul
PC principale.

## Nota importante

Non dire che le maschere sono ground truth. Le maschere sono output dei
modelli, soprattutto SAM2 o AnomalyVFM. La ground truth usata nelle metriche
principali è composta da box CVAT.
