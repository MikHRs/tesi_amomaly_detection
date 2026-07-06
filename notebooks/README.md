# Notebook

I notebook sono opzionali e servono soltanto per esplorazione e figure.
La pipeline riproducibile resta negli script in `src/`.

Se si aggiunge un notebook:

1. usare percorsi relativi alla radice del progetto;
2. non incorporare frame o video sensibili nel file;
3. annotare versioni di modello, prompt e configurazione;
4. spostare il codice riutilizzabile in un modulo Python.

## Stato dei notebook correnti

- `E01_SAM2_v2.ipynb`: E01-v2 SAM2 con box manuale, eseguito;
- `E02_Grounded_SAM2_v2.ipynb`: E02-v2 Grounding DINO periodico + SAM2,
  eseguito;
- `E05_AnomalyDINO_v1.ipynb`: E05-v1 AnomalyDINO 16-shot, eseguito;
- `E06_AnomalyVFM_v1.ipynb`: E06-v1 AnomalyVFM DINOv2 zero-shot, eseguito
  su Kaggle;
- `E07_AnomalyVFM_External_v1.ipynb`: sorgente E07-v1 sul benchmark termico
  esterno; Kaggle lo ha scaricato senza output incorporati, conservati
  separatamente nel pacchetto E07 sotto `results/anomaly_vfm/`;
- `E08_AnomalyDINO_External_v1.ipynb`: E08 AnomalyDINO sul benchmark termico
  esterno, eseguito su Kaggle con output incorporati;
- `E09_AnomalyDINO_SAM2_v1.ipynb`: E09 raffinamento automatico
  AnomalyDINO → punto → SAM2, eseguito su Kaggle con output incorporati;
- `E10_AnomalyDINO_MultiPoint_SAM2_v1.ipynb`: E10 multi-candidato
  AnomalyDINO → SAM2, versione corretta memory-safe e riproducibile;
- `E10_AnomalyDINO_MultiPoint_SAM2_v1_executed.ipynb`: notebook Kaggle
  eseguito, con output e celle di recupero dopo il riavvio RAM;
- `sam2_pilot_colab.ipynb` e `grounded_sam2_pilot_colab.ipynb`: versioni
  storiche dei pilot E01–E03;
- `e04_one_class_anomaly_colab.ipynb`: preparato ma non eseguito e mantenuto
  come pilot storico.

E04 non è il protocollo finale: usa validation normale anche nella
valutazione e non adotta il terzo video come sorgente separata di normalità.
I notebook AnomalyDINO e AnomalyVFM seguono
`docs/notes/piano_anomaly_dino_vfm.md` e conservano gli output in cartelle
distinte sotto `results/`.
