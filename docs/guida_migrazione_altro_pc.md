# Guida rapida: usare il progetto su un altro PC

## Idea generale

Il progetto è diviso in due parti:

1. repository Git: codice, notebook, documentazione, configurazioni e test;
2. archivio separato: `data/` e `results/`, cioè video, frame, export CVAT,
   dataset processati e risultati pesanti.

Questa separazione evita di caricare su GitHub materiale pesante o sensibile.

## Sul PC principale

Dalla cartella del progetto:

```bash
cd /home/michelerais/Documenti/TESI/tesi_anomaly_detection
```

Verificare lo stato:

```bash
git status
```

Creare un archivio di dati e risultati:

```bash
cd /home/michelerais/Documenti/TESI
tar --exclude='tesi_anomaly_detection/.venv' \
    --exclude='tesi_anomaly_detection/.git' \
    -czf tesi_anomaly_detection_data_results_2026-07-06.tar.gz \
    tesi_anomaly_detection/data \
    tesi_anomaly_detection/results
```

Calcolare l'hash:

```bash
sha256sum tesi_anomaly_detection_data_results_2026-07-06.tar.gz
```

## Sul nuovo PC

Clonare la repository:

```bash
git clone <URL_REPOSITORY_PRIVATA>
cd tesi_anomaly_detection
```

Creare l'ambiente Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Eseguire i test:

```bash
python -m unittest discover -s tests -v
```

Ripristinare dati e risultati, partendo dalla cartella che contiene
l'archivio:

```bash
tar -xzf tesi_anomaly_detection_data_results_2026-07-06.tar.gz
```

Se l'archivio viene estratto dentro la stessa cartella che contiene
`tesi_anomaly_detection/`, le sottocartelle `data/` e `results/` tornano al
posto giusto.

## Quando apri Codex sull'altro PC

Aprire la cartella:

```text
tesi_anomaly_detection/
```

Poi chiedere a Codex:

```text
Leggi CONTEXT_FOR_CODEX.md e riprendi il progetto da lì.
```

## Cosa non va messo su GitHub

Non caricare pubblicamente:

- video originali del relatore;
- export CVAT sensibili;
- frame estratti;
- checkpoint pesanti;
- dataset processati se contengono materiale non pubblicabile;
- zip dei risultati con frame/video.

Per questo `data/`, `results/`, video, checkpoint e zip sono esclusi da Git
tramite `.gitignore`.
