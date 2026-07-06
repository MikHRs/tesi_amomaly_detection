# Guida E10 su Kaggle

## File

- notebook:
  `notebooks/E10_AnomalyDINO_MultiPoint_SAM2_v1.ipynb`;
- input già usato da E09:
  `data/colab_uploads/E09_sam2_refinement_inputs_v1.zip`;
- protocollo congelato:
  `docs/protocollo_e10_anomalydino_multipoint_sam2.md`.

## Preparazione

1. Aprire un nuovo notebook Kaggle e importare il file `.ipynb`.
2. Attivare una GPU T4. Il codice usa `cuda:0`; una seconda T4 non è
   necessaria.
3. Attivare Internet per clonare SAM2 e scaricare il checkpoint.
4. Con `Add Input`, collegare lo stesso dataset privato usato per E09.
   In alternativa caricare
   `E09_sam2_refinement_inputs_v1.zip` come nuovo dataset privato.
5. Verificare che sotto `/kaggle/input` sia presente
   `E09_inputs/protocol.json`.

## Esecuzione cella per cella

1. **Verifica input.** L'output atteso contiene:
   `File verificati: 34`, `CUDA: True` e il nome della GPU.
2. **Installazione SAM2.** Devono comparire:
   commit `2b90b9f...` e checksum checkpoint `6d1aa6f...`.
3. **Caricamento e protocollo.** Devono comparire:
   500 righe anomale, 20 normali, 3.732 box persona, 1.473 box `incerto`,
   1.000 box `incerto_sui_binari` e `K: (1, 3, 5)`.
4. **Probe.** Il primo punto deve coincidere con E09; devono apparire fino a
   cinque box colorate e le box CVAT verdi. Non modificare i parametri dopo
   aver visto la figura.
5. **Inferenza completa.** Il notebook elabora 500 frame e i 20 controlli.
   Al termine scaricare subito `E10_inference_checkpoint.zip`.
6. **Metriche.** Controllare che la tabella contenga 30 righe:
   cinque scope × tre valori di K × due soglie IoU. Le righe K=1 dello
   scope `persona` devono essere molto vicine ai risultati E09. Lo scope
   decisivo per gli oggetti è `incerto_sui_binari`.
7. **Figure.** Controllare il grafico precision/recall/F1 e il confronto
   qualitativo K=1 contro K=5.
8. **Pacchetto finale.** Scaricare
   `E10_v1_anomalydino_multipoint_sam2_results.zip`.
9. Salvare una versione del notebook con gli output e scaricare anche il
   `.ipynb`.

## Cosa inviare qui

Inviare, in quest'ordine:

1. screenshot della tabella `Metriche E10`;
2. `E10_v1_anomalydino_multipoint_sam2_results.zip`;
3. notebook `.ipynb` scaricato da Kaggle.

Non scegliere soltanto il K con risultato migliore: E10 deve conservare e
discutere K=1, K=3 e K=5.
