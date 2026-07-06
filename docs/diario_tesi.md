# Diario della tesi

Copiare il blocco seguente per ogni sessione di lavoro.

```text
Data:
Obiettivo:
Cosa ho fatto:
File prodotti:
Problemi:
Domande per il relatore:
Prossimo passo:
```

Annotare anche, quando rilevante: video/clip usati, versione dell'export,
prompt, soglia, modello/checkpoint, hardware e tempo impiegato.

---

## 2026-07-06 — E10 multi-candidato completato

**Obiettivo:** superare il limite dichiarato di E09, che produce al massimo
una maschera per frame pur avendo più persone visibili.

**Protocollo fissato:** mappe e gate E05 congelati; primo punto identico a
E09; fino a cinque massimi separati di `64 px`; SAM 2.1 Small senza
fine-tuning; configurazioni annidate K=1,3,5; deduplicazione box a IoU `0,7`;
matching uno-a-uno a IoU `0,3` e `0,5`, separatamente per `persona`,
`incerto_sui_binari`, tutti gli `incerto`, tutte le anomalie sui binari e
tutte le annotazioni visibili.

**Controlli locali:** notebook JSON e sintassi validi; K=1 replica le
coordinate E09 su tutti i 500 frame; distanza minima verificata per i cinque
punti su tutti i frame; test sintetico del matching superato.

**Correzione di scope prima dell'inferenza:** l'audit ha rilevato che il
massimo E05 colpisce una persona in 478/500 frame, ma un'annotazione
`incerto` in 0/500. Anche le proposte top 1% E05 hanno 0 match a IoU 0,3
contro le 1.473 box `incerto`, delle quali 1.000 appartengono a due track
`sui_binari`. E10 deve quindi misurare esplicitamente questi oggetti.

**Risultati:** sulle persone K=3 offre il migliore compromesso a IoU 0,3
(precision `0,844`, recall `0,298`, F1 `0,440`), rispetto a F1 `0,231` di
K=1. K=5 porta il recall a `0,318` ma riduce F1 a `0,412`. Per le 1.000 box
`incerto_sui_binari`, tutte le configurazioni producono `0` TP sia a IoU 0,3
sia a IoU 0,5. Il limite non dipende quindi soltanto dal massimo globale.

**Runtime:** 2.520 maschere in `121,06 s`, `20,816` maschere/s e picco GPU
`0,61 GB`. I quattro falsi positivi normali di frame restano invariati.

**Problema e recupero:** la prima figura qualitativa ha esaurito la RAM
conservando maschere booleane non compresse. Il checkpoint era già stato
scaricato. La figura è stata rigenerata dalle box; il notebook riproducibile
è stato poi corretto per liberare le maschere dopo il packbits.

**File prodotti:** notebook memory-safe
`notebooks/E10_AnomalyDINO_MultiPoint_SAM2_v1.ipynb`, notebook Kaggle
eseguito `notebooks/E10_AnomalyDINO_MultiPoint_SAM2_v1_executed.ipynb`,
protocollo, checkpoint e pacchetto finale sotto
`results/sam_refinement/E10_v1_anomalydino_multipoint_sam2/`.

**Conclusione:** più candidati migliorano la copertura delle persone ma non
recuperano gli oggetti incerti sui binari. Il prossimo passo scientifico va
concordato con il relatore: detector specifico, revisione semantica delle
track incerto oppure chiusura esplicita con questo limite.

## 2026-07-06 — Relazione completa per il ricevimento

**Obiettivo:** presentare al relatore tutto il lavoro, distinguendo
esperimenti, attività sui dati, riproducibilità, risultati e limiti.

**File prodotti:**

- `docs/Relazione_avanzamento_Michele_Rais_2026-07-07.pdf`, 24 pagine,
  versione compatta da 1,3 MB;
- versione ad alta qualità nella stessa cartella;
- sorgente HTML riproducibile;
- `docs/guida_ricevimento_2026-07-07.md`, copione di 10–15 minuti;
- cartella pronta all'uso `docs/ricevimento_2026-07-07/`, con PDF, due video
  dimostrativi e figure E09.

**Contenuto:** otto esperimenti eseguiti E01, E02, E03, E05, E06, E07, E08
ed E09; E04 esplicitamente indicato come protocollo storico preparato ma non
eseguito. Sono incluse anche revisione CVAT, gestione dei dati, benchmark
esterno, analisi degli errori, commit, checksum e decisioni richieste.

**Prossimo passo:** ricevimento del 7 luglio, conferma del perimetro e avvio
della stesura della tesi.

## 2026-07-06 — E09 raffinamento AnomalyDINO con SAM2

**Obiettivo:** verificare quantitativamente se SAM2 migliora la
localizzazione di un candidato proposto automaticamente da AnomalyDINO.

**Protocollo:** massimo della mappa E05 come punto positivo; SAM 2.1 Small
image predictor senza fine-tuning; 500 frame San Donato; confronto della box
SAM2 e della componente AnomalyDINO contenente il punto con le box persona
CVAT v2. I 20 normali di test E05 costituiscono il controllo negativo.

**Risultati:** IoU media da `0,501` a `0,717`; IoU mediana da `0,506` a
`0,757`; hit rate a IoU 0,5 da `0,534` a `0,934`. SAM2 migliora 457/500
frame e il guadagno medio è `+0,215`. Le 504 maschere richiedono `98,02 s`
su Tesla T4.

**Limiti:** un solo punto produce al massimo una maschera e non recupera
tutte le 3.732 persone annotate. Quando il punto cade sulla regione
sbagliata SAM2 rifinisce l'oggetto sbagliato. I quattro falsi positivi normali
di E05 restano quattro dopo SAM2.

**Interpretazione:** SAM2 è efficace come rifinitore geometrico
condizionato, ma non come correttore della decisione di anomalia. La qualità
del prompt automatico rimane determinante.

**File prodotti:** risultati consolidati in
`results/sam_refinement/E09_v1_anomalydino_point_sam2/`; SHA256 pacchetto
finale
`93fc9d4aec71d022b0bad76dfbce4a16be8d0bacb55ff174a9011f2e375bed0c`.
Notebook eseguito in `notebooks/E09_AnomalyDINO_SAM2_v1.ipynb`, SHA256
`476a8f62168c6eb614a9727972d95e2767bcc70e3daa20d2aaf8a3fd5cb823d9`,
con 14 output incorporati e nessun errore.

**Prossimo passo:** chiudere la campagna sperimentale principale e produrre
la relazione tecnica aggiornata per il relatore, quindi iniziare la stesura
della tesi.

## 2026-07-05 — E08 AnomalyDINO sul benchmark termico esterno

**Obiettivo:** confrontare AnomalyDINO con E07 sul medesimo benchmark,
misurando anche l'effetto del numero di riferimenti normali.

**Protocollo:** 200 clip normali per memorie annidate da 16, 64 e 200
riferimenti; 100 clip normali disgiunti per calibrare tre soglie al
percentile 95; seed `20260705`; test ufficiale di 3.509 frame. Nessun clip è
condiviso tra riferimento, calibrazione e test. Nessun fine-tuning.

**Risultati:** il 16-shot ottiene il migliore ranking globale, AUROC `0,593`
e AP `0,400`. Il 200-shot ottiene il migliore punto operativo, con precision
`0,418`, recall `0,153` e F1 `0,224`, e la migliore AUROC media per clip
`0,621`. Con 200 riferimenti e smoothing 11, l'F1 degli eventi a tIoU 0,3 è
`0,263` (5 TP, 4 FP e 24 FN).

**Interpretazione:** AnomalyDINO supera AnomalyVFM E07 nel ranking globale,
ma la separazione e soprattutto il recall restano insufficienti. Il
comportamento non è monotono rispetto al numero di riferimenti: più
riferimenti migliorano il confronto entro ciascun clip e il punto operativo,
ma non l'AUROC globale.

**File prodotti:** checkpoint verificato e analisi locale in
`results/anomaly_dino/E08_v1_anomalydino_external_nested_shots/`. SHA256 del
checkpoint base:
`8de98304bf0cadcaa0d72317c9b68eefaded2fda0037893b65285008028cd32d`.
Il pacchetto consolidato ha SHA256
`675d1cde8a69ab92989813c9bc0871b0a98ddbbc4202e336e93a197cafb35fb0`.
Il notebook Kaggle eseguito è archiviato in
`notebooks/E08_AnomalyDINO_External_v1.ipynb`, SHA256
`807fb79884ed932f67ffa90047a23bafcaa89949b2f65f06511840012dbe0378`,
con 26 output incorporati e nessun errore di esecuzione.

**Analisi qualitativa:** conservati due veri positivi, due falsi positivi e
due falsi negativi su clip distinti. Le mappe rispondono frequentemente a
veicoli, luci e strutture calde, confermando che salienza termica e anomalia
semantica non coincidono necessariamente.

**Prossimo passo:** confronto conclusivo E07–E08 e decisione sul test E09 di
raffinamento SAM2, limitandolo a candidati automatici verificabili.

## 2026-07-05 — E07 AnomalyVFM sul benchmark termico esterno

**Obiettivo:** verificare la generalizzazione di AnomalyVFM in un benchmark
in cui normali e anomalie compaiono negli stessi clip, riducendo il cambio
globale di scena presente in E06.

**Dataset:** Thermal Anomaly Detection Dataset versione 2, 300 clip normali
di training e 29 clip test. Il test contiene 3.509 frame: 2.385 normali e
1.124 anomali, con annotazioni temporali binarie.

**Protocollo:** stesso checkpoint AnomalyVFM DINOv2 di E06, nessun
fine-tuning, due Tesla T4, batch 24. Valutazione con soglia nativa `0,5` e
con soglia `0,465039`, percentile 95 di un frame centrale per ciascuno dei
300 clip normali.

**Risultati:** AUROC `0,462` e AP `0,309`, sotto la prevalenza positiva
`0,320`. La soglia nativa ottiene F1 `0,099`; la calibrazione porta F1 a
`0,164`, con 1.000 falsi negativi. Soltanto 13 clip su 29 hanno AUROC
superiore a 0,5.

**Analisi temporale:** lo smoothing a 11 frame porta AUROC soltanto a
`0,465`. A tIoU 0,3 riduce le predizioni a 17 eventi, ma ne abbina 3 su 29:
precision `0,176`, recall `0,103`, F1 `0,130`.

**Interpretazione:** il modello risponde spesso a sorgenti calde statiche e
non rappresenta adeguatamente le anomalie semantiche/comportamentali del
porto. La calibrazione modifica la soglia, ma non può correggere un ranking
prossimo al caso casuale. E07 ridimensiona quindi la lettura dei risultati
perfetti di E06.

**File prodotti:** checkpoint e pacchetto finale in
`results/anomaly_vfm/E07_v1_anomalyvfm_external_zero_shot/`. SHA256 del
pacchetto finale:
`9e50514c0fb6a3aa4f9709dbd0c53d5636bcb6d71e20c26a12fa87332cf3d74f`.
Il notebook sorgente è archiviato in
`notebooks/E07_AnomalyVFM_External_v1.ipynb`, SHA256
`52e73eb73d7c06f96af5a9430327654edf7a2f9b0c0a242ebcdf9e2d0d38d34b`.
Kaggle lo ha esportato senza output incorporati; gli artefatti eseguiti
restano nel pacchetto finale.

**Prossimo passo:** E08 AnomalyDINO sullo stesso split esterno, usando
riferimenti normali del training ufficiale.

## 2026-07-05 — E06 AnomalyVFM DINOv2 zero-shot

**Obiettivo:** valutare AnomalyVFM senza riferimenti normali, calibrazione o
fine-tuning sui frame termici ferroviari e confrontarlo con AnomalyDINO E05.

**Protocollo:** checkpoint `MaticFuc/anomalyvfm_dinov2`, revisione
`c7698cefc6415ca43c93c3e924e02f6863c57e73`, codice ufficiale al commit
`4da96b493a0b12e4e380fa5bf6573c126342855c`. Test su 20 frame normali
indipendenti e 500 frame della clip San Donato, GPU Tesla T4 su Kaggle.

**Risultati di frame:** con soglia nativa `0,5`, TN 20, FP 0, FN 0 e TP 500;
precision, recall, F1, AUROC e AP pari a `1,000`. Inferenza in `702,48 s`,
circa `0,74 frame/s`.

**Risultati di localizzazione:** nessuna maschera nativa supera la soglia
`0,5`. Il massimo della mappa cade su una persona in soli `7/500` frame
(`0,014`) e su una qualsiasi annotazione visibile in `65/500` (`0,130`).
Con componenti dal top 1% adattivo, F1 `0,201` a IoU 0,3 e `0,071` a IoU
0,5.

**Interpretazione:** la testa di score distingue nettamente i due video, ma
la mappa risponde soprattutto a massicciata, bordi e altri elementi della
scena. La classificazione perfetta non equivale quindi a localizzazione
corretta e può dipendere in larga parte dal domain shift tra i due video.

**File prodotti:** archivio verificato e contenuto estratto in
`results/anomaly_vfm/E06_v1_anomalyvfm_dinov2_zero_shot/`. Lo ZIP ha
SHA256 `9175e8f87e2c2a1db6b5bf74e5150020e70ca8be0c46f9c9d8a56afe7dd3d864`.
Il notebook Kaggle è archiviato in `notebooks/E06_AnomalyVFM_v1.ipynb`,
SHA256 `40476a0bae6cf1d9446d9190ab8b20d2655d7de8fc5a3d5a0eb761d0a352034c`.

**Prossimo passo:** ripetere il confronto sul Thermal Anomaly Detection
Dataset esterno, dove lo split ufficiale riduce la confusione tra anomalia e
cambio di scena.

## 2026-07-05 — E05 AnomalyDINO 16-shot

**Obiettivo:** verificare se una memoria di feature DINOv2 costruita da frame
termici normali consente di distinguere e localizzare i frame anomali.

**Protocollo:** 16 riferimenti normali, 20 normali di calibrazione, 20 normali
di test e 500 frame anomali San Donato. La soglia di frame è stata scelta
soltanto sui normali di calibrazione. Nessun fine-tuning, rotazione o
mascheramento PCA.

**Risultati di frame:** TN 16, FP 4, FN 0, TP 500; precision `0,992`, recall
`1,000`, F1 `0,996`, AUROC e AP `1,000`. Velocità `12,07 frame/s`.

**Risultati di localizzazione:** pointing game persona `478/500 = 0,956`.
Con box dal top 1% adattivo, a IoU 0,3: precision `0,745`, recall `0,242`, F1
`0,365`; almeno un match in tutti i frame. La soglia spaziale assoluta
calibrata sui normali fallisce sotto il cambio globale di scena.

**Analisi errori:** i quattro falsi positivi sono frame normali consecutivi
183–186, con risposta sul ballast e sul bordo destro. I 22 fallimenti del
pointing game includono diversi massimi vicini ai bordi delle box.

**Limite principale:** normali e anomalie provengono da scene differenti.
La separazione perfetta degli score può includere un forte contributo di
domain shift; i 500 frame anomali sono inoltre temporalmente correlati.

**File prodotti:** pacchetto e contenuto verificato in
`results/anomaly_dino/E05_v1_anomalydino_16shot/`; notebook eseguito in
`notebooks/E05_AnomalyDINO_v1.ipynb`.

**Prossimo passo:** archiviare il notebook Colab, quindi eseguire AnomalyVFM
zero-shot sul medesimo test congelato.

## 2026-07-04 — Aggiornamento del piano dopo incontro con il relatore

**Obiettivo:** riallineare repository, dominio e piano sperimentale alle
indicazioni dell'ultimo incontro.

**Cosa ho fatto:**

- generalizzato il titolo da solo ferroviario a immagini/video termici;
- mantenuto i video ferroviari come caso di studio;
- separato i ruoli di CVAT, SAM/SAM2, AnomalyDINO e AnomalyVFM;
- identificato `ir_2025-01-31_04-15-15.mp4` come sorgente normale da
  verificare;
- preparato script per frame normali, frame test, mask→box e tabella di
  revisione;
- documentato che AnomalyDINO usa riferimenti normali mentre AnomalyVFM è il
  confronto zero-shot;
- marcato E04 come pilot storico non adatto alla valutazione finale;
- letto criticamente i PDF e i link forniti dal relatore;
- rigenerato l'inventario con i percorsi correnti: tre video leggibili su tre;
- verificati gli script su frame reali in un ambiente temporaneo.

**Risultati sperimentali:** nessun nuovo risultato. AnomalyDINO, AnomalyVFM e
il raffinamento automatico con SAM non sono ancora stati eseguiti.

**Problemi:** l'ambiente `.venv` locale è incoerente e deve essere ricreato.
L'inventario con percorsi obsoleti è stato rigenerato.

**Prossimo passo:** congelare gli intervalli per riferimento, validation e
test e preparare l'esperimento AnomalyDINO.

### Ambiente e frame normali

- ambiente `.venv` ricreato e dipendenze installate;
- 16 test automatici superati;
- video `ir_2025-01-31_04-15-15.mp4` verificato come normale nel contesto
  concordato;
- estratti 200 frame distanziati tra 0 e 298,96 secondi in
  `data/normal_train_frames/`;
- controllato un campione di 12 frame distribuiti sull'intera durata.

### Revisione CVAT San Donato

Ricevuto ed archiviato l'export
`test_video_san_donato_v2_reviewed.zip` in formato CVAT for video 1.1.

Controlli eseguiti:

- task corretto, 6575 frame e risoluzione 640×512;
- 20 track e 59.650 box interpolate;
- unica label `anomalia`;
- nessuna box con coordinate invalide o fuori immagine;
- controllo visivo campionato, incluso l'intervallo 98–101 secondi;
- export e CSV precedenti conservati come versione v1.

La conversione CSV conserva ora gli attributi `tipo`, `visibilita` e
`posizione`. Nella v2 risultano 14 track `persona` e 6 track `incerto`; queste
ultime devono essere confermate dal relatore oppure escluse dalla metrica
principale e analizzate separatamente.

La ground truth canonica in
`results/annotations/test_video_san_donato_boxes.csv` ora corrisponde alla
v2 revisionata. Il ricalcolo E01-v2 è riportato nella sezione successiva,
mentre E02 resta da aggiornare.

### E01-v2 — ripetizione SAM2 con box manuale

E01 è stato ripetuto su Google Colab, GPU Tesla T4, usando SAM 2.1 Small e lo
stesso prompt `[392, 195, 438, 270]` al primo frame della clip 85–105 secondi.
Sono state propagate 500 maschere.

Valutazione sulla track CVAT v2 `0`, 490 frame comuni:

- IoU media `0,755`;
- IoU mediana `0,769`;
- IoU minima `0,561`;
- F1 `1,000` sia a IoU 0,3 sia a IoU 0,5.

Il risultato è valido come valutazione di segmentazione/tracking condizionata
da una box manuale accurata, non come anomaly detection autonoma. Pacchetto
archiviato in `results/qualitative/sam2_pilot_01_v2/` e notebook eseguito
salvato in `notebooks/E01_SAM2_v2.ipynb`.

### E02-v2 — Grounding DINO periodico e inserimento di nuove identità

E02 è stato ripetuto sulla stessa clip con Grounding DINO Tiny, prompt
`person`, e SAM 2.1 Small. Il detector è stato eseguito ogni 50 frame. Le due
persone presenti all'inizio sono state inizializzate come ID 1 e 2; una
persona entrata successivamente è stata rilevata al frame clip 250 e aggiunta
dinamicamente come ID 3.

Confronto box su ground truth CVAT v2:

- ID 1 / track 10: IoU media `0,675`, F1@0,5 `0,888`;
- ID 2 / track 0: IoU media `0,751`, F1@0,5 `1,000`;
- ID 3 / track 12: IoU media `0,684`, F1@0,5 `0,959`;
- aggregato: 1232 frame comuni, IoU media `0,707`, F1@0,5 `0,946`.

Non risultano frame GT mancanti sulle tre associazioni. Al frame 350 una
quarta box grezza è una proposta duplicata, non una quarta persona. Il
pacchetto verificato è archiviato in
`results/qualitative/grounded_sam2_pilot_02_v2/` e il notebook eseguito in
`notebooks/E02_Grounded_SAM2_v2.ipynb`.

Il test resta una baseline semantica `person` più tracking: non assegna un
anomaly score rispetto alla normalità.

## 2026-07-03 — Annotazione e primo pilot SAM2

**Obiettivo:** creare una prima ground truth e verificare SAM2 zero-shot su
un breve segmento termico ferroviario.

**Cosa ho fatto:**

- annotati in CVAT Online due video con label `anomalia` e attributi;
- esportati i task in formato `CVAT for video 1.1`;
- controllati XML, track, box e interpolazioni;
- preparata una clip di San Donato tra 85 e 105 secondi;
- eseguito SAM 2.1 Small su Google Colab con GPU Tesla T4;
- fornita una box manuale al frame iniziale;
- propagata la maschera per 500 frame senza fine-tuning;
- salvati video, maschere, box, metadati e confronto con CVAT.

**File prodotti:**

- `data/exports_cvat/ir_2025-01-31_05-09-12.mp4.zip`;
- `data/exports_cvat/test_video_san_donato.mp4.zip`;
- `results/qualitative/sam2_pilot_01/`;
- `results/metrics/sam2_pilot_01/`;
- `notebooks/sam2_pilot_colab.ipynb`.

**Risultato osservato:** nei frame campionati SAM2 mantiene la stessa persona
e segue i cambiamenti di postura. Il confronto box è provvisorio: IoU media
0,587; F1 0,833 a IoU 0,3 e 0,714 a IoU 0,5.

**Problemi:** una parte della track CVAT è interpolata a destra della persona,
quindi le metriche sottostimano visibilmente SAM2. Sul primo frame era inoltre
presente un piccolo componente spurio della maschera.

**Domande per il relatore:** confermare che il pilot box-prompted sia incluso
come esperimento di segmentazione/tracking assistito e che il passo successivo
sia la proposta automatica di box tramite prompt testuale.

**Prossimo passo:** correggere la ground truth attorno a 98–101 secondi e
provare `prompt testuale → Grounding DINO → box → SAM2`.

### Estensione E02 — Prompt testuale

È stato eseguito anche il flusso `person → Grounding DINO → box → SAM2`.
Grounding DINO ha trovato due persone reali nel frame iniziale con score
0,771 e 0,631; una rilevazione debole da 0,22 è stata eliminata fissando la
soglia di selezione a 0,30. SAM2 ha seguito le due persone per 500 frame.

È emerso un limite operativo utile per la tesi: una terza persona entra
successivamente e non può essere scoperta se il detector viene eseguito solo
al frame iniziale. È inoltre emersa una lacuna nella ground truth: una delle
persone rilevate è visibile per 273 frame prima dell'inizio della sua track
CVAT.

### Estensione E03 — Confronto dei prompt

Sullo stesso frame iniziale sono stati confrontati i prompt `person`, `box`,
`pallet`, `obstacle`, `foreign object` e `anomaly`, usando le medesime soglie.
`person` ha selezionato correttamente le due persone visibili. `box` ha
selezionato tre piccoli elementi rettangolari. I prompt generici `obstacle` e
`foreign object` hanno generato più candidati, ma anche regioni eterogenee.
Il prompt `anomaly` ha selezionato soltanto la persona centrale e un oggetto,
scartando la persona al bordo sinistro alla soglia 0,30.

**Conclusione:** Grounding DINO è sensibile alla formulazione del prompt, ma
questo test riguarda soltanto la proposta di regioni. Grounding DINO e SAM2
restano moduli ausiliari: la tesi deve valutare anomaly detection/localization,
assegnando un'anomaly score rispetto alla normalità e al contesto, non
limitandosi a riconoscere categorie di oggetti.

**Limite:** il confronto è su un solo frame e senza ground truth separata per
classe; i conteggi sono descrittivi e non metriche definitive.
