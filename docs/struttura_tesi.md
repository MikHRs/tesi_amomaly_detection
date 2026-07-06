# Struttura proposta della tesi

## 1. Problema

### 1.1 Immagini termiche e motivazione

Vantaggi e limiti delle camere termiche nei diversi domini applicativi. La
sicurezza della sede ferroviaria è il caso di studio sperimentale.

### 1.2 Definizione del compito

Localizzazione class-agnostic della label `anomalia`; distinzione tra object
detection, open-vocabulary detection e anomaly detection one-class.

### 1.3 Domande di ricerca

- Metodi zero-shot o reference-based costruiti su feature prevalentemente RGB
  localizzano anomalie in immagini termiche?
- Come distinguere una vera decisione di anomalia dalla semplice
  identificazione di una categoria di oggetto?
- Quanto incidono dimensione, visibilità, contrasto termico e posizione?
- SAM/SAM2 può rifinire una regione proposta da una anomaly map?

### 1.4 Vincoli

Pochi dati, domain shift RGB→termico, correlazione temporale, riservatezza e
assenza di fine-tuning sui video ferroviari nel nucleo della tesi.

## 2. Tecnologie usate

### 2.1 Video e immagini termiche

Proprietà, vantaggi notturni, limiti e domain shift rispetto a RGB.

### 2.2 Annotazione con CVAT

Box, track, keyframe, `outside`, `occluded`, export XML e controllo qualità.

### 2.3 SAM e SAM2

Segmentazione promptable, prompt geometrici e propagazione temporale.

### 2.4 AnomalyDINO e AnomalyVFM

Feature DINOv2, memoria di patch normali, differenza tra reference-based
training-free e zero-shot con pesi adattati su dati ausiliari.

### 2.5 Zero-shot e open-vocabulary

Prompt testuali, eventuale detector testo-box e differenza rispetto a SAM.

### 2.6 Metriche

AUROC/AP a livello di frame, IoU, pointing game, matching uno-a-uno, TP, FP,
FN, precision, recall e F1. Le metriche pixel-level richiedono maschere.

## 3. Design del sistema

### 3.1 Requisiti e architettura

Pipeline locale: inventario, estrazione di riferimenti normali e test,
annotazione, anomaly localization, raffinamento SAM, valutazione e produzione
dei risultati.

### 3.2 Organizzazione e trattamento dei dati

Originali, copie di lavoro, naming, privacy, versionamento degli export.

### 3.3 Protocollo di annotazione

Label unica, attributi, regole di inclusione/esclusione e revisione.

### 3.4 Strategie di localizzazione e prompt

Anomaly map AnomalyDINO/AnomalyVFM, conversione in box/punti e collegamento
con SAM/SAM2. Grounding DINO resta una baseline ausiliaria.

### 3.5 Formati e riproducibilità

CSV delle box, configurazioni YAML, struttura dei risultati e comandi CLI.

## 4. Esperimenti

### 4.1 Dataset e protocollo sperimentale

Video normale separato, due video anomali annotati, campionamento, statistiche
e separazione temporale tra memoria, validation e test.

### 4.2 SAM prompt-based

Box manuale e box Grounding DINO, segmentazione e tracking, limiti del prompt
iniziale.

### 4.3 AnomalyDINO / AnomalyVFM

Confronto tra metodo reference-based e metodo zero-shot, con configurazioni,
checkpoint, soglie e hardware registrati.

### 4.4 Raffinamento con SAM

Anomaly map → box/punto → SAM; confronto della localizzazione grezza e
raffinata su un sottoinsieme congelato.

### 4.5 Risultati quantitativi

Metriche globali e, se i dati bastano, per tipo, visibilità e posizione.

### 4.6 Analisi qualitativa degli errori

Esempi di TP, FP e FN; piccoli oggetti, basso contrasto, occlusioni e sfondo.

### 4.7 Limiti e sviluppi futuri

Dimensione del test set, domain shift, dipendenza dai prompt, possibili dati
aggiuntivi e fine-tuning futuro.
