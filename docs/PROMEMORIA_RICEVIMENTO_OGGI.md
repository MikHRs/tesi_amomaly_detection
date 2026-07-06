# Promemoria ricevimento: cosa mostrare e cosa dire

## Obiettivo della riunione

Chiedere al professore se la direzione va bene e quale scope scegliere per il
prossimo passo:

1. concentrarsi sulle persone come anomalie in area ferroviaria;
2. oppure affrontare esplicitamente gli oggetti/ostacoli piccoli sui binari,
   dove E10 ha mostrato un limite.

## Apertura da dire

> Professore, rispetto alla scorsa volta ho organizzato la repository e ho
> provato una pipeline completa. Il lavoro non è object detection pura: sto
> trattando anomaly detection/localization. Ho separato tre problemi: capire se
> il frame è anomalo, proporre dove guardare, e segmentare/localizzare la
> regione candidata.

## Cosa mostrare, in ordine

### 1. Repository e cartelle

Aprire:

```text
README.md
```

Dire:

> Ho strutturato la repo separando codice, dati, notebook, risultati e
> documentazione. `src/` contiene gli script riproducibili, `notebooks/` gli
> esperimenti E01-E10, `results/` gli output e `docs/` protocolli e relazioni.

Se chiede dei dati:

> I video e i frame pesanti non sono caricati su Git. Sono in `data/` e
> ripristinabili da archivio separato, mentre su Git tengo codice,
> configurazioni, notebook e documentazione.

### 2. Come sono fatte le annotazioni CVAT

Aprire:

```text
results/annotations/test_video_san_donato_v2_reviewed_boxes.csv
```

Dire:

> La ground truth principale non è una maschera pixel-level. È fatta di box
> CVAT. Ogni riga è una box in un frame, con coordinate `xtl, ytl, xbr, ybr`,
> `track_id`, label `anomalia` e attributi come `tipo`, `visibilita`,
> `posizione`.

Frase chiave:

> Le maschere che mostro non sono ground truth: sono maschere predette dai
> modelli, soprattutto SAM2. Io le confronto con le box CVAT tramite IoU.

### 3. E10: esperimento più importante

Aprire:

```text
results/sam_refinement/E10_v1_anomalydino_multipoint_sam2/export/E10_qualitative_k1_vs_k5.png
```

Dire:

> Qui AnomalyDINO genera una heatmap di anomalia. Prendo i massimi locali
> della heatmap e li uso come punti positivi per SAM2. SAM2 genera maschere,
> poi trasformo le maschere in box e le confronto con CVAT.

Parametri da dire:

```text
K = 1, 3, 5 punti candidati
distanza minima tra punti = 64 px
deduplicazione box = IoU 0.7
metriche = IoU 0.3 e 0.5
500 frame anomali
20 frame normali di controllo
2520 maschere SAM2 generate
```

### 4. Risultati numerici E10

Aprire:

```text
results/sam_refinement/E10_v1_anomalydino_multipoint_sam2/export/E10_detection_summary.csv
```

Dire:

> Sulle persone, K=3 è il miglior compromesso: a IoU 0.3 passa a precision
> 0.844, recall 0.298, F1 0.440. K=5 aumenta poco il recall ma peggiora la
> precisione.

Poi dire il limite:

> Sugli oggetti `incerto_sui_binari`, invece, il risultato è zero true
> positive. Quindi il metodo recupera soprattutto persone o regioni
> termicamente evidenti, ma non ancora gli oggetti piccoli/incerti sui binari.

### 5. Conclusione da proporre al prof

Dire:

> La mia interpretazione è: SAM2 funziona bene come raffinamento quando il
> punto candidato è giusto, ma il collo di bottiglia è la proposta del
> candidato. AnomalyDINO tende a puntare sulle persone o su regioni calde, non
> necessariamente sugli ostacoli piccoli sui binari.

Domanda finale:

> Secondo lei conviene restringere la tesi alle persone come anomalie in area
> ferroviaria, oppure impostare il prossimo esperimento su ostacoli/binari con
> una strategia dedicata, ad esempio ROI dei binari o selezione dei candidati
> vincolata alla zona ferroviaria?

## Se il professore chiede "che maschere hai usato?"

Risposta:

> Ground truth: box CVAT, non maschere. Maschere predette: SAM2 in E01, E02,
> E09, E10; AnomalyVFM produce mappe/mask a bassa risoluzione in E06/E07.
> Per le metriche principali trasformo le maschere SAM2 in box e confronto con
> le box CVAT tramite IoU.

## Se chiede "che impostazioni?"

Risposta breve:

> In E10 uso AnomalyDINO 16-shot come generatore di heatmap, poi seleziono
> K=1,3,5 massimi locali separati di 64 px. Ogni punto entra in SAM2 come
> positive point. Le maschere vengono deduplicate con IoU 0.7 e valutate con
> IoU 0.3 e 0.5 contro CVAT.

## Se chiede "come hai costruito il database?"

Risposta:

> Non è un database SQL, è un dataset su filesystem con manifest CSV. I video
> originali stanno in `data/raw`, gli export CVAT in `data/exports_cvat`, le
> annotazioni convertite in `results/annotations`, i dataset processati in
> `data/processed`, e gli output sperimentali in `results`. I notebook non sono
> la fonte unica: gli script in `src` rendono riproducibili le conversioni e le
> metriche.

## Frase finale forte

> Il risultato principale finora non è "ho risolto tutto", ma ho capito dove
> funziona e dove no: SAM2 migliora molto la localizzazione del candidato, ma
> il candidato automatico deve essere migliorato, soprattutto per oggetti
> piccoli sui binari.
