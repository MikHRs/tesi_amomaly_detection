# Guida breve per il ricevimento del 7 luglio 2026

## Prima dell'incontro

1. Apri `00_Relazione_CHIARA.pdf`.
2. Usa la visualizzazione a pagina intera.
3. Tieni chiusi i video E01/E02: mostrali soltanto se il professore li chiede.
4. La relazione tecnica di 24 pagine è un allegato, non il documento da
   presentare.

## Apertura

> «Professore, prima di mostrarle i risultati voglio distinguere chiaramente
> ciò che è stato davvero valutato. Le prove migliori riguardano soprattutto
> le persone. Ho quindi eseguito un test con fino a cinque candidati e
> metriche separate: migliora la copertura delle persone, ma gli oggetti
> annotati come “incerto” sui binari restano a zero true positive.»

## Scaletta da 8–10 minuti

### Pagine 1–2 — Risposta e annotazioni

- sì: persone riconosciute, segmentate e seguite;
- no: oggetti `incerto` sui binari non localizzati dal protocollo testato;
- nella clip: 3.732 box persona e 1.473 incerto;
- due track `incerto_sui_binari`, ID 9 e 18, producono 1.000 box.

### Pagina 3 — Scope reale degli esperimenti

- E01, E02 ed E09 valutano persone;
- E03 è soltanto un confronto qualitativo di prompt;
- E05/E06 separano decisione di frame e localizzazione;
- E07/E08 hanno soltanto ground truth temporale.

### Pagina 4 — Risultati solidi

- E01: IoU media persona 0,755;
- E02: tre persone, IoU media 0,707 e F1@0,5 0,946;
- E05: massimo su una persona in 478/500 frame;
- E09: IoU persona 0,501 → 0,717.

### Pagina 5 — Lacuna

- massimo AnomalyDINO: 478 frame persona, 0 incerto, 22 nessuna box;
- proposte E05 a IoU 0,3: 902 TP persona, 0 TP incerto;
- non dire che gli ostacoli generici siano già riconosciuti.

### Pagina 6 — E10

- proposte annidate K=1,3,5;
- metriche separate per persona e incerto sui binari;
- K=3 porta F1 persona da 0,231 a 0,440 a IoU 0,3;
- tutte le configurazioni restano a 0 TP sugli incerti sui binari.

### Pagina 7 — Decisioni richieste

1. Rinominare in CVAT le track incerto 9 e 18 con un tipo più preciso?
2. Restringere la tesi alle persone in area ferroviaria o mantenere
   l'obiettivo di anomalia generica?
3. Dato il risultato E10 sugli oggetti, provare un detector dedicato o
   chiudere con questo limite?

## Frase conclusiva

> «Il risultato più solido è che SAM2 rifinisce molto bene un candidato
> corretto. E10 mostra però che AnomalyDINO propone soprattutto persone:
> aumentare i candidati ne migliora il recall, ma non recupera gli oggetti
> incerti sui binari.»
