# Guida per il ricevimento del 7 luglio 2026

## Prima dell'incontro

1. Apri la cartella `docs/ricevimento_2026-07-07/`.
2. Apri `00_Relazione_avanzamento.pdf` e attiva la visualizzazione a pagina
   intera.
3. Tieni pronti, ma chiusi:
   - `01_E01_SAM2_box_manuale.mp4`;
   - `02_E02_GroundingDINO_SAM2.mp4`.
4. Non aprire Kaggle, Colab o il codice salvo domanda specifica.
5. Il documento ha 24 pagine ma non va letto tutto: serve come supporto e
   come materiale da lasciare al professore.

## Apertura consigliata

> «Professore, dopo il nostro ultimo incontro ho organizzato il lavoro in una
> pipeline completa. Ho revisionato la ground truth, confrontato SAM2,
> Grounding DINO, AnomalyDINO e AnomalyVFM, aggiunto un benchmark termico
> esterno e chiuso con un test automatico AnomalyDINO-punto-SAM2. Vorrei
> mostrarle prima il quadro complessivo, poi tre risultati chiave e infine
> confermare con lei il nucleo definitivo della tesi.»

## Copione da 10–15 minuti

### 0:00–1:00 — Obiettivo e lavoro complessivo

Mostra le pagine 2–3.

Da dire:

- il compito è localizzazione class-agnostic di `anomalia`;
- E01–E09 sono identificativi;
- gli esperimenti eseguiti sono otto;
- E04 è rimasto un pilot storico perché il protocollo successivo è più
  rigoroso;
- oltre agli esperimenti ci sono revisione CVAT, repository, test,
  riproducibilità e analisi degli errori.

### 1:00–2:00 — Dati e ground truth

Mostra brevemente le pagine 5–6.

Da dire:

- tre video ferroviari verificati;
- clip principale San Donato 85–105 s, 500 frame;
- ground truth CVAT v2 revisionata;
- la revisione ha cambiato l'IoU E01 da 0,587 a 0,755;
- benchmark esterno: 29 clip e 3.509 frame, ma solo ground truth temporale.

### 2:00–3:30 — SAM2 e Grounding DINO

Mostra le pagine 9–10. Se il professore vuole, apri per 10–15 secondi i due
video.

Messaggi:

- E01: SAM2 traccia bene una persona quando riceve una box corretta;
- E02: Grounding DINO trova le persone e, rieseguito ogni 50 frame, consente
  di aggiungere una nuova identità;
- risultato E02-v2: IoU media 0,707 e F1@0,5 0,946;
- limite: riconoscere `person` non equivale a decidere `anomalia`.

### 3:30–5:30 — Confronto E05–E06

Mostra le pagine 12–13.

Da dire:

- E05 AnomalyDINO è reference-based e usa 16 normali;
- E06 AnomalyVFM è zero-shot e non usa i riferimenti;
- entrambi separano perfettamente i due video a livello di score;
- E05 localizza spesso le persone, pointing game 0,956;
- E06 ha pointing game 0,014;
- conclusione: classificazione del frame e localizzazione sono problemi
  distinti;
- i risultati perfetti possono dipendere dal cambio di scena.

### 5:30–7:30 — Controllo esterno E07–E08

Mostra le pagine 14–16.

Da dire:

- normali e anomalie sono ora nello stesso clip;
- AnomalyVFM scende ad AUROC 0,462;
- AnomalyDINO raggiunge AUROC 0,593 nel 16-shot;
- con 200 riferimenti ottiene il miglior F1, 0,224;
- il benchmark ridimensiona correttamente i risultati perfetti del caso
  ferroviario;
- le mappe rispondono spesso a salienza termica, non alla semantica
  dell'evento.

### 7:30–9:30 — Risultato finale E09

Mostra le pagine 17–19.

Da dire:

- il punto non è manuale: è il massimo della mappa AnomalyDINO;
- SAM2 riceve quel punto e rifinisce la regione;
- IoU media: 0,501 → 0,717;
- hit rate a IoU 0,5: 0,534 → 0,934;
- 457 frame su 500 migliorano;
- se il punto è sbagliato, SAM2 segmenta bene l'oggetto sbagliato;
- i quattro falsi positivi normali restano quattro.

Frase chiave:

> «SAM2 risolve la qualità geometrica del candidato, non la decisione
> semantica di anomalia.»

### 9:30–11:00 — Conclusione scientifica

Mostra pagina 20.

> «Il sistema più credibile è modulare: un anomaly detector produce score e
> candidati, SAM2 li rifinisce, CVAT consente la valutazione. Il limite
> principale resta la qualità del candidato iniziale e il domain shift.»

### 11:00–13:00 — Decisioni richieste

Mostra pagina 21 e chiedi esplicitamente:

1. conferma del titolo;
2. conferma che E04 non debba essere recuperato;
3. conferma che il fine-tuning resti sviluppo futuro;
4. conferma dell'uso del benchmark esterno;
5. trattamento delle track `incerto`;
6. autorizzazione delle immagini nelle slide;
7. durata e formato della discussione.

## Se il professore interrompe o il tempo è poco

Versione da cinque minuti:

1. pagina 3: contributi e tabella dei risultati;
2. pagina 8: mappa E01–E09;
3. pagine 12–14: perché i risultati perfetti non generalizzano;
4. pagina 17: miglioramento E09;
5. pagina 21: decisioni richieste.

## Cose da non dire

- Non dire «SAM rileva le anomalie»: SAM segmenta un prompt.
- Non dire «AnomalyDINO è zero-shot puro»: usa riferimenti normali.
- Non dire «abbiamo addestrato SAM2»: non è stato fatto fine-tuning.
- Non dire «il sistema è perfetto»: E05/E06 hanno domain shift.
- Non presentare AUROC 1,000 senza citare il cambio di scena.
- Non dire che E09 recupera tutte le persone: usa un solo punto per frame.
- Non confondere score di frame, localizzazione box e maschera.

## Se chiede “perché non fare fine-tuning?”

Risposta:

> «Il nucleo della tesi confronta zero-shot, reference-based e prompt-based.
> Con pochi video correlati, un fine-tuning adesso rischierebbe leakage e una
> validazione poco credibile. Preferirei indicarlo come sviluppo futuro, a
> meno che lei non ritenga indispensabile una prova molto limitata.»

## Se chiede “qual è il risultato migliore?”

Risposta:

> «Il risultato più solido è E09: a parità di punto automatico, SAM2 porta
> l'IoU media da 0,501 a 0,717 su 500 frame. Il risultato più importante dal
> punto di vista metodologico è invece E06–E08: una buona classificazione di
> frame non garantisce localizzazione né generalizzazione.»

## Chiusura

> «Se confermiamo oggi titolo, perimetro e assenza di nuovi esperimenti
> pesanti, passo subito alla stesura dei capitoli e le invio una prima versione
> completa per la revisione.»
