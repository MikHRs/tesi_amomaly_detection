# Ultimo incontro con il relatore

Questo documento riordina le decisioni emerse dalla trascrizione. Non
costituisce un risultato sperimentale.

## Decisione sul titolo

Il titolo non deve limitare la tesi al solo contesto ferroviario. La
formulazione di lavoro è:

> Localizzazione di anomalie in immagini termiche mediante foundation model
> visuali e segmentazione prompt-based

Una formulazione alternativa, più prudente, è:

> Valutazione preliminare di metodi zero-shot e prompt-based per la
> localizzazione di anomalie in immagini termiche

I video ferroviari restano il caso di studio principale. Dataset termici
esterni possono essere aggiunti solo quando la definizione di anomalia e la
ground truth sono esplicite.

## Ruolo di CVAT

CVAT resta necessario per costruire e revisionare la ground truth, stimare
falsi positivi e falsi negativi e controllare la localizzazione. La label
principale rimane `anomalia`.

Persona, pallet, scatola o altre categorie non devono trasformare il task in
object detection multi-classe. Eventuali attributi descrivono visibilità,
posizione, contrasto termico o incertezza, ma non cambiano la label usata
nella valutazione principale. Non è necessario riaprire tutte le vecchie
annotazioni soltanto per uniformare attributi secondari.

## Ruolo di SAM

Le prove SAM/SAM2 già eseguite restano parte della tesi. Mostrano il ruolo
della segmentazione prompt-based nel dominio termico:

- SAM/SAM2 riceve un punto, una box o una maschera;
- segmenta o propaga l'oggetto indicato;
- non decide autonomamente se la regione sia anomala;
- può rifinire una regione proposta da un anomaly detector.

Google Colab con GPU T4 è sufficiente per i pilot correnti; non è necessario
acquistare GPU o noleggiare altre risorse per questa fase.

## Nuova direzione: AnomalyDINO / AnomalyVFM

La pipeline deve includere almeno un metodo che produca direttamente un
anomaly score e una anomaly map.

È importante distinguere i due metodi:

- **AnomalyDINO** è training-free ma reference-based: confronta feature
  DINOv2 del test con patch estratte da immagini normali del dominio. I circa
  200 frame normali servono a questo esperimento.
- **AnomalyVFM** è un metodo zero-shot con pesi già adattati su dati
  ausiliari: in inferenza non costruisce la normalità dai 200 frame
  ferroviari. Può quindi essere valutato come confronto zero-shot.

Anomalib può offrire un'interfaccia comune, ma versione, checkpoint e comandi
devono essere registrati durante l'esecuzione reale.

## Uso del video normale

Il video non annotato e privo di anomalie evidenti è:

```text
ir_2025-01-31_04-15-15.mp4
```

Prima di usarlo come normalità occorre guardarlo integralmente o almeno
effettuare un controllo sistematico. Da questo video si estraggono circa 200
frame temporalmente distanziati per la memoria normale di AnomalyDINO.

I due video già annotati restano il test anomalo:

```text
ir_2025-01-31_05-09-12.mp4
test_video_san_donato.mp4
```

Una porzione normale indipendente deve restare fuori dalla memoria di
AnomalyDINO per misurare i falsi positivi. Il vecchio dataset E04, costruito
con frame normali e anomali provenienti dagli stessi due video, rimane un
pilot storico e non rappresenta il protocollo finale.

## Esperimenti da fare

1. **SAM prompt-based:** conservare e organizzare i pilot già eseguiti,
   esplicitando il tipo di prompt e il fatto che SAM non assegna l'anomaly
   score.
2. **AnomalyDINO / AnomalyVFM:** produrre score, mappe, maschere e overlay su
   frame termici, senza inventare risultati prima dell'esecuzione.
3. **Raffinamento con SAM:** ricavare punti o box da alcune anomaly map e
   confrontare la localizzazione prima e dopo SAM.

## Falsi positivi e falsi negativi

Un falso negativo è un'anomalia reale non rilevata; un falso positivo è una
regione normale segnalata come anomala. Nel caso di sicurezza ferroviaria i
falsi negativi sono più critici, mentre i falsi positivi possono essere
revisionati da un operatore.

La scelta di una soglia non deve basarsi sui frame del test finale. Filtri
come una dimensione minima possono ridurre il rumore, ma rischiano di
eliminare anomalie piccole o lontane e devono quindi essere valutati, non
assunti corretti.

## Dataset alternativi

Il *Thermal Anomaly Detection Dataset* di Kaggle contiene sequenze normali di
training, sequenze anomale di test e annotazioni temporali dei frame anomali.
È adatto a una verifica di video anomaly detection a livello di frame, ma non
fornisce automaticamente la stessa ground truth spaziale delle box CVAT.

Dataset di semantic segmentation termica sono utili per discutere il dominio
e il problema dei bordi, ma non diventano benchmark di anomaly detection
senza una definizione esplicita di normalità e anomalia.

## Prossimi passi

1. ricreare l'ambiente Python; l'inventario è già stato rigenerato con i
   percorsi validi;
2. controllare il video normale e congelare gli intervalli di riferimento e
   di test normale;
3. estrarre i 200 frame normali e i frame test annotati;
4. eseguire AnomalyDINO su Colab e conservare tutti i parametri;
5. eseguire separatamente AnomalyVFM zero-shot;
6. convertire alcune mappe in box e provarle come prompt SAM;
7. confrontare output e CVAT, dando priorità al recall e all'analisi dei
   falsi negativi.
