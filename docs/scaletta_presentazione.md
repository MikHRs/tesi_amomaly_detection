# Scaletta per le slide

Proposta per una presentazione di circa 8–10 minuti. Ogni slide deve avere un
messaggio principale e poco testo.

## 1. Titolo e obiettivo

**Messaggio:** valutare preliminarmente la localizzazione di anomalie in
immagini termiche mediante foundation model visuali.

- titolo della tesi;
- nome, relatore e corso;
- una sola immagine termica rappresentativa, se autorizzata.

## 2. Problema

**Messaggio:** persone e oggetti estranei possono costituire anomalie, ma il
dominio termico è diverso dalle immagini RGB comuni.

- caso di studio ferroviario: persone, scatole, pallet, legno;
- label principale unica: `anomalia`;
- vincoli: pochi dati, video, dominio termico e domain shift.

## 3. Dati e ground truth

**Messaggio:** CVAT fornisce la risposta corretta contro cui valutare i
modelli.

- tre video ricevuti;
- due video annotati nel pilot;
- box track, attributi, `occluded` e `outside`;
- breve spiegazione dell'IoU.

## 4. Pipeline

```text
video → CVAT → ground truth
video normale → AnomalyDINO → anomaly map
frame test → AnomalyVFM zero-shot → anomaly map
anomaly map → box/punto → SAM2 → raffinamento
ground truth + predizioni → metriche e analisi errori
```

Il pilot ausiliario già eseguito è:

```text
prompt testuale → Grounding DINO → box → SAM2 → maschera/track
```

## 5. Ruolo di SAM2

**Messaggio:** SAM2 segmenta e segue un candidato, ma non decide da solo che
cosa sia anomalo.

- input: punto, box o maschera;
- output: maschera propagata nel video;
- nessun fine-tuning nel pilot.

## 6. Esperimento E01

- clip San Donato 85–105 secondi;
- SAM 2.1 Small;
- Tesla T4 su Google Colab;
- una box al frame iniziale;
- 500 frame propagati.

## 7. Risultato qualitativo

Usare:

`results/qualitative/sam2_pilot_01_v2/six_frames.png`

Mostrare eventualmente pochi secondi del video overlay. Evidenziare che
l'identità viene mantenuta anche quando cambia la postura e sono presenti
altre persone.

## 8. Risultati quantitativi E01-v2

- IoU media: 0,755;
- IoU mediana: 0,769;
- F1 a IoU 0,3: 1,000;
- F1 a IoU 0,5: 1,000;
- 490 frame comuni con la ground truth v2.

Scrivere chiaramente `tracking condizionato da box manuale; non anomaly
detection autonoma`.

## 9. Analisi critica

Confrontare brevemente le metriche v1 e v2.

**Messaggio:** la revisione della ground truth ha portato l'IoU media da
0,587 a 0,755 e ha eliminato i fallimenti sotto IoU 0,5. La qualità delle
annotazioni modifica sostanzialmente la valutazione.

## 10. Confronto E05–E06

- 16 riferimenti normali, 20 normali di calibrazione e 20 di test;
- 500 frame anomali San Donato;
- F1 di frame `0,996`, nessun falso negativo e quattro falsi positivi;
- pointing game `0,956`;
- box top 1% a IoU 0,3: precision `0,745`, recall `0,242`;
- mostrare `results/anomaly_dino/E05_v1_anomalydino_16shot/export/score_distribution.png`
  e `qualitative_six_frames.png`.

**Messaggio critico:** l'AUROC perfetta non prova da sola una soluzione
generale, perché i video normale e anomalo mostrano scene differenti. La
soglia spaziale assoluta fallisce sotto questo domain shift.

Per E06 AnomalyVFM:

- vero zero-shot, nessun riferimento normale o calibrazione;
- F1 di frame `1,000`, ma zero maschere native non vuote;
- pointing game persona `0,014`;
- box top 1% a IoU 0,3: precision `0,281`, recall `0,156`, F1 `0,201`;
- mostrare `results/anomaly_vfm/E06_v1_anomalyvfm_dinov2_zero_shot/export/E06_score_distribution.png`
  e `E06_anomalyvfm_qualitative_six_frames.png`.

**Messaggio:** classificare correttamente un frame e localizzare
correttamente l'anomalia sono due obiettivi distinti.

## 11. E07–E08 — controllo sul benchmark esterno

- 29 clip con normali e anomalie nella stessa scena;
- 3.509 frame con ground truth temporale;
- AUROC AnomalyVFM `0,462`;
- F1 nativo `0,099`, calibrato `0,164`;
- AUROC AnomalyDINO `0,593` con 16 riferimenti;
- miglior F1 AnomalyDINO `0,224` con 200 riferimenti;
- smoothing temporale insufficiente per entrambi;
- mostrare `results/anomaly_vfm/E07_v1_anomalyvfm_external_zero_shot/export/E07_quantitative_overview.png`.
- affiancare
  `results/anomaly_dino/E08_v1_anomalydino_external_nested_shots/export/E08_quantitative_overview.png`.

**Messaggio:** il controllo esterno ridimensiona il risultato perfetto E06;
AnomalyDINO migliora il ranking rispetto ad AnomalyVFM, ma non raggiunge un
recall sufficiente e dipende dal numero di riferimenti.

## 12. Piano sperimentale e conclusione

- mostrare il pilot completato Grounding DINO + SAM2 con prompt `person`;
- presentare E03 soltanto come analisi del modulo ausiliario che propone
  regioni, non come risultato finale di anomaly detection;
- evidenziare due persone iniziali e una terza persona aggiunta automaticamente
  con detection ogni 50 frame, senza fine-tuning;
- riportare per E02-v2 IoU media aggregata `0,707` e F1@0,5 `0,946` su 1232
  frame comuni delle tre track abbinate;
- discutere la proposta duplicata al frame 350 e l'occlusione finale;
- AnomalyDINO reference-based e AnomalyVFM zero-shot completati;
- ripetere il confronto sul benchmark termico esterno;
- convertire alcune anomaly map in prompt per SAM2;
- valutare in modo esplicito falsi positivi e falsi negativi;
- correzione ground truth;
- nessun risultato inventato e limiti dichiarati.

Conclusione provvisoria: SAM2 ha mostrato un tracking termico promettente
quando riceve una box accurata. Grounding DINO ha fornito automaticamente le
box dal prompt `person`; con detection periodica è stato possibile aggiungere
una nuova identità al frame 250. Il confronto E03 indica inoltre che un
detector testuale può proporre candidati, ma non stabilisce se siano anomalie.
AnomalyDINO ha prodotto regioni candidate senza prompt manuale, con buona
capacità di trovare almeno una persona ma localizzazione box grossolana e
forte sensibilità al cambio di scena. AnomalyVFM separa nettamente i due
video a livello di score, ma le sue mappe non localizzano in modo affidabile
le persone. La fase successiva userà il benchmark termico esterno e valuterà
SAM2 soltanto quando il candidato automatico è sufficientemente accurato.

Aggiornamento E09 da inserire nella conclusione:

- punto automatico dal massimo AnomalyDINO, nessun prompt manuale;
- IoU media box `0,501 → 0,717`;
- hit rate a IoU 0,5 `0,534 → 0,934`;
- 457/500 frame migliorati;
- quattro falsi positivi normali invariati;
- messaggio: SAM2 rifinisce bene il candidato, ma non decide se sia anomalo
  e non corregge un prompt errato.
