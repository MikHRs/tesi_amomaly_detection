# Frasi per la tesi aggiornate

Le frasi sono bozze da adattare allo stile del capitolo e ai risultati
realmente ottenuti.

## Obiettivo

> L'obiettivo della tesi è valutare in modo preliminare metodi zero-shot e
> prompt-based per la localizzazione di anomalie in immagini termiche,
> combinando strumenti di annotazione manuale, modelli di anomaly
> localization basati su foundation model e segmentazione prompt-based.

## Caso di studio

> La parte sperimentale principale considera video termici in ambito
> ferroviario; tale dominio è trattato come caso di studio di un problema più
> generale di localizzazione di anomalie in immagini termiche.

## CVAT

> Le annotazioni CVAT costituiscono una ground truth preliminare
> class-agnostic, nella quale la label principale è `anomalia` e gli eventuali
> attributi descrivono proprietà secondarie quali posizione, visibilità e
> incertezza.

## SAM

> SAM non viene utilizzato come sistema autonomo di anomaly detection, ma come
> strumento di segmentazione prompt-based per rifinire o analizzare regioni
> candidate.

> Nel raffinamento automatico E09, il massimo della mappa AnomalyDINO è stato
> usato come punto positivo per SAM2. L'IoU media della box è aumentata da
> 0,501 a 0,717 e l'hit rate a IoU 0,5 da 0,534 a 0,934. Il miglioramento
> riguarda la geometria del candidato selezionato: SAM2 non corregge un punto
> posto sulla regione sbagliata e non modifica la decisione di anomalia a
> livello di frame.

## AnomalyVFM e AnomalyDINO

> A differenza di SAM, metodi come AnomalyDINO o AnomalyVFM mirano a produrre
> direttamente mappe di anomalia, risultando quindi più vicini al problema
> della localizzazione automatica di anomalie.

> AnomalyDINO è un metodo training-free basato sul confronto tra feature di
> patch DINOv2 e riferimenti normali del dominio; quando utilizza frame
> ferroviari normali non viene pertanto considerato zero-shot puro.

> AnomalyVFM viene invece valutato in modalità zero-shot, impiegando pesi
> pre-addestrati senza adattamento sui frame termici ferroviari.

> Nel test ferroviario, AnomalyVFM separa completamente gli score dei frame
> normali e anomali usando la soglia nativa del modello; tuttavia le mappe
> non localizzano in modo affidabile le persone, ottenendo un pointing game
> pari a 0,014. Tale risultato evidenzia che un'elevata accuratezza a livello
> di frame non implica una corretta localizzazione spaziale dell'anomalia.

> Il confronto con AnomalyDINO mostra un compromesso opposto: AnomalyDINO,
> pur richiedendo riferimenti normali del dominio, concentra più spesso i
> massimi sulle persone, mentre AnomalyVFM zero-shot risponde soprattutto a
> elementi dello sfondo. Entrambi i risultati restano condizionati dal
> cambio di scena tra il video normale e quello anomalo.

> Nel benchmark termico esterno, nel quale intervalli normali e anomali
> appartengono agli stessi clip, AnomalyVFM ottiene AUROC pari a 0,462. La
> calibrazione della soglia su frame normali migliora il recall, ma non il
> ranking degli score. Il risultato suggerisce che la separazione perfetta
> osservata tra i due video ferroviari fosse influenzata dal cambiamento
> globale di scena e non costituisse evidenza sufficiente di
> generalizzazione.

> Sullo stesso benchmark esterno, AnomalyDINO reference-based raggiunge
> AUROC globale pari a 0,593 con 16 riferimenti, superando AnomalyVFM ma
> restando lontano da una separazione affidabile. Con 200 riferimenti il
> ranking globale scende a 0,578, mentre F1 alla soglia calibrata e AUROC
> media per clip salgono rispettivamente a 0,224 e 0,621. Il numero di
> riferimenti modifica quindi sia la rappresentazione della normalità sia
> la calibrazione degli score tra scene, senza un vantaggio monotono su
> tutte le metriche.

## Falsi negativi

> Nel contesto considerato, i falsi negativi risultano particolarmente critici
> perché corrispondono ad anomalie reali non rilevate dal sistema, mentre i
> falsi positivi possono essere successivamente verificati da un operatore.

## Limiti

> I risultati devono essere interpretati come una valutazione preliminare:
> il numero limitato di video, la correlazione temporale tra frame e il
> cambiamento di dominio da immagini RGB a immagini termiche non consentono
> di trarre conclusioni sull'impiego operativo del sistema.
