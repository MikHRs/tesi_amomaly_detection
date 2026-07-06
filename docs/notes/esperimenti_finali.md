# Esperimenti finali

Questa è una bozza del capitolo sperimentale. Gli stati distinguono risultati
già ottenuti da attività ancora da eseguire.

## Esperimento 1 - SAM prompt-based

**Stato:** pilot eseguito; metriche ancora provvisorie per la revisione della
ground truth.

Obiettivo: valutare la capacità di SAM2 di segmentare e seguire in video
termici una regione indicata con box manuale o proposta da Grounding DINO.

Materiale già disponibile:

- E01: SAM2 con box manuale;
- E02: Grounding DINO con prompt `person` seguito da SAM2;
- E03: sensibilità di Grounding DINO alla formulazione del prompt.

Interpretazione corretta: l'esperimento misura segmentazione/tracking
prompt-based. Non dimostra anomaly detection autonoma. Le metriche definitive
vanno ricalcolate dopo la revisione CVAT.

## Esperimento 2 - AnomalyDINO / AnomalyVFM

**Stato:** AnomalyDINO E05-v1, AnomalyVFM E06-v1/E07-v1 ed E08
AnomalyDINO esterno completati quantitativamente e archiviati. Per E08
restano le figure qualitative e il notebook finale.

Obiettivo: ottenere anomaly score e mappe senza un prompt manuale sul singolo
oggetto.

Confronti:

- AnomalyDINO con circa 200 frame normali termici come riferimenti;
- AnomalyVFM zero-shot senza adattamento sui frame ferroviari;
- opzionalmente, baseline E04 PatchCore-style come controllo interno.

Output da conservare:

- score per frame;
- anomaly map grezza;
- maschera sogliata;
- overlay;
- configurazione completa e tempi.

Il test deve contenere normali indipendenti e anomalie annotate. La soglia non
deve essere scelta sui frame test anomali.

E06 usa la soglia nativa `0,5` di AnomalyVFM senza calibrazione sul test.
Ottiene F1 di frame `1,000`, ma zero maschere native non vuote e pointing
game persona `0,014`. Il top 1% adattivo raggiunge F1 box `0,201` a IoU 0,3.
Il confronto con E05 mostra quindi una netta separazione tra capacità di
classificare il video e capacità di localizzare l'anomalia.

E07 valuta lo stesso checkpoint in 29 clip esterni con normali e anomalie
nella stessa scena. AUROC scende a `0,462`; F1 è `0,099` con soglia nativa e
`0,164` con calibrazione normale. Questo controllo mostra che il risultato
perfetto di E06 non generalizza al benchmark semantico e rafforza l'ipotesi
del domain shift.

E08 valuta AnomalyDINO sul medesimo benchmark, con memorie annidate da 16,
64 e 200 riferimenti. Il miglior ranking globale è il 16-shot, con AUROC
`0,593`; il miglior F1 alla soglia calibrata è invece il 200-shot, `0,224`.
L'AUROC media per clip del 200-shot è `0,621`. AnomalyDINO supera quindi
AnomalyVFM E07, ma mantiene un recall basso e non risolve il problema.

## Esperimento 3 - Raffinamento con SAM

**Stato:** E09 completato quantitativamente e qualitativamente.

Obiettivo: verificare su pochi frame se una box o un punto derivati dalla
anomaly map consentono a SAM/SAM2 di migliorare la localizzazione.

Confrontare:

1. anomaly map/maschera originale;
2. box candidata estratta automaticamente;
3. maschera raffinata da SAM;
4. ground truth CVAT.

Riportare anche i fallimenti. Se il candidato iniziale è assente o sulla
regione sbagliata, SAM non deve essere descritto come soluzione del falso
negativo.

E09 usa il massimo della mappa AnomalyDINO E05 come punto positivo per SAM
2.1 Small. Sui 500 frame San Donato l'IoU media della box passa da `0,501`
a `0,717` e l'hit rate a IoU 0,5 da `0,534` a `0,934`; 457 frame migliorano.
Il risultato dimostra che SAM2 rifinisce efficacemente un candidato valido.
Non modifica però i quattro falsi positivi normali e non recupera gli altri
oggetti presenti nello stesso frame: un singolo punto produce al massimo una
maschera.

## Valutazione falsi positivi/falsi negativi

Per ogni metodo compilare:

- TP: anomalia presente e rilevata;
- FN: anomalia presente ma non rilevata;
- FP: regione o frame normale segnalato;
- qualità della localizzazione;
- dimensione e visibilità dell'anomalia;
- note sul contesto.

Nel caso ferroviario, il recall e i falsi negativi hanno priorità
interpretativa. Questo non giustifica però una soglia arbitrariamente bassa:
il costo in falsi positivi deve essere mostrato.

## Discussione

La discussione dovrà separare:

- capacità di assegnare un anomaly score;
- capacità di localizzare grossolanamente;
- capacità di segmentare un candidato già indicato;
- trasferimento da immagini RGB generiche a immagini termiche;
- vantaggi e limiti del contesto temporale video.

Le conclusioni devono restare formulate come valutazione preliminare.
AnomalyDINO e AnomalyVFM sono stati eseguiti e registrati, ma il cambio di
scena impedisce di attribuire la separazione perfetta esclusivamente alle
anomalie. AnomalyVFM, in particolare, separa i frame ma non localizza in
modo affidabile le persone. Non affermare che il raffinamento con SAM
risolva il problema se il candidato iniziale è assente o scorretto.
