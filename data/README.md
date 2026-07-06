# Dati

Questa cartella separa gli originali dai prodotti intermedi:

- `raw/`: video originali ricevuti dal relatore; non modificarli;
- `work/`: copie di lavoro e dataset esterni;
- `frames/`: frame estratti;
- `processed/`: dataset derivati riproducibili per gli esperimenti;
- `clips/`: clip brevi per il pilot CVAT;
- `exports_cvat/`: export XML o ZIP di CVAT;
- `normal_train_frames/`: riferimenti normali per AnomalyDINO;
- `test_anomaly_frames/`: frame test selezionati dai video annotati;
- `cvat_annotations/`: copie di lavoro delle annotazioni usate nei nuovi
  esperimenti;
- `sam_outputs/`: maschere/overlay prodotti da SAM o SAM2;
- `anomaly_outputs/`: score e mappe importati dagli ambienti di inferenza.

Tutte queste sottocartelle sono escluse da Git. Non caricare materiale del
relatore su servizi cloud senza autorizzazione esplicita.

## Dataset termico esterno opzionale

Il dataset Kaggle *Thermal Anomaly Detection Dataset* può essere collocato,
manualmente e nel rispetto della licenza, in:

```text
data/work/external/thermal_anomaly_detection/
```

È un benchmark di videosorveglianza termica open-set derivato da
*Seasons in Drift*: contiene 300 clip normali di training, 29 clip anomale di
test e annotazioni temporali dei frame anomali. Non è ferroviario e non
sostituisce il test set della tesi. Può servire per un controllo esterno a
livello di frame. Le annotazioni temporali non vanno trattate come box o
maschere di localizzazione senza una verifica concreta dei file.
