# Protocollo E10 — AnomalyDINO multi-punto → SAM2

## Stato

Protocollo fissato prima dell'esecuzione. E10 estende E09 senza modificare
mappe, score, soglia di frame, modello SAM2 o annotazioni.

## Domanda sperimentale

E09 usa il solo massimo globale della mappa AnomalyDINO e può quindi produrre
al massimo una maschera per frame. Nei 500 frame di San Donato sono invece
visibili 5.205 box: 3.732 `persona` e 1.473 `incerto`. Tra queste ultime,
1.000 box appartengono a due track marcate `sui_binari`. E10 misura se un
numero limitato di massimi spazialmente separati aumenta la copertura sia
delle persone sia degli oggetti incerti sui binari, e quale costo introduce
in termini di falsi positivi e tempo.

## Input congelati

- mappe, score e split di `E05_v1_anomalydino_16shot`;
- clip San Donato di 500 frame, frame sorgente 2125–2624;
- 20 frame normali indipendenti di E05 come controllo negativo;
- box CVAT v2 revisionate, con attributi `tipo` e `posizione`;
- soglia di frame E05 `0,28902236819267274`;
- SAM 2.1 Small, commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- checkpoint SHA256
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.

Non sono ammessi fine-tuning, prompt manuali o regolazioni basate sulle box di
test.

## Generazione delle proposte

Per ogni frame che supera la soglia di anomalia congelata:

1. il primo punto è il massimo globale già congelato da E05/E09, così `K=1`
   replica esattamente E09;
2. la mappa AnomalyDINO viene ridimensionata a `640×512`;
3. viene soppressa un'area circolare di raggio `64 px`;
4. il procedimento viene ripetuto sulla mappa soppressa fino a ottenere
   cinque punti;
5. ogni punto è passato indipendentemente a SAM2 come prompt positivo;
6. per ogni punto si conserva la maschera con maggiore IoU predetta da SAM2;
7. le box vengono deduplicate in ordine di rango del punto, rimuovendo una
   nuova box se ha IoU almeno `0,7` con una box già conservata.

Le configurazioni valutate sono annidate:

- `K=1`, controllo che replica E09;
- `K=3`;
- `K=5`.

Vengono riportate tutte le configurazioni; non viene selezionato a posteriori
il valore di `K` migliore.

## Valutazione

Le predizioni e le box `persona` sono abbinate uno-a-uno, separatamente per
frame, con matching greedy ordinato per IoU decrescente. Le soglie sono
`IoU≥0,3` e `IoU≥0,5`.

Le metriche vengono calcolate separatamente per cinque scope:

- `persona`;
- `incerto_all`;
- `incerto_sui_binari`;
- `all_sui_binari`;
- `all_visible`.

Per ogni scope, valore di `K` e soglia si riportano:

- numero di predizioni e box ground truth;
- true positive, false positive e false negative;
- precision, recall e F1;
- IoU media delle coppie abbinate;
- frame con almeno una corrispondenza;
- proposte medie per frame;
- tempo, maschere al secondo e picco di memoria GPU.

Questa estensione è stata aggiunta prima dell'inferenza E10 dopo avere
verificato che le metriche E01–E09 erano centrate soprattutto sulle persone:
il massimo E05 colpiva `persona` in 478/500 frame e `incerto` in 0/500.

Il controllo sui 20 frame normali conserva il gate di E05. SAM2 viene eseguito
soltanto sui frame già classificati come anomali; si riportano sia i falsi
positivi di frame, che non possono essere corretti da SAM2, sia il numero di
proposte spaziali prodotte.

## Interpretazione prevista

E10 non misura una segmentazione semantica completa: la ground truth è
costituita da box, non da maschere. L'esperimento misura il compromesso tra
copertura multi-oggetto e precisione. Un aumento del recall accompagnato da
una riduzione della precisione è un risultato informativo, non un errore del
protocollo.
