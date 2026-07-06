"""Nota eseguibile sulla pipeline AnomalyVFM/AnomalyDINO → SAM.

AnomalyVFM è un metodo zero-shot: usa un modello già adattato su dati
ausiliari e non costruisce la normalità dai frame ferroviari. AnomalyDINO è
invece reference-based e training-free: confronta le feature DINOv2 dei frame
test con una memoria di patch ricavata da immagini normali del dominio.

Input e output previsti:

* AnomalyVFM: immagine test → anomaly score + anomaly map/mask;
* AnomalyDINO: immagini normali + immagine test → score + anomaly map;
* SAM/SAM2: immagine + punto/box/maschera candidata → segmentazione raffinata.

Pseudo-pipeline:

1. eseguire AnomalyVFM oppure AnomalyDINO in un ambiente Colab separato;
2. salvare score e anomaly map senza sovrascriverli;
3. sogliare la mappa e convertirla in box candidate;
4. fornire box o punti a SAM/SAM2;
5. confrontare output grezzo e raffinato con la ground truth CVAT.

Questo modulo non scarica checkpoint e non esegue modelli pesanti. Per una
prova reale usare Anomalib o i repository ufficiali e registrare versione,
checkpoint, soglie, hardware e comandi.
"""

from __future__ import annotations


def main() -> int:
    print("AnomalyVFM produce anomaly score + anomaly mask.")
    print("SAM invece richiede un prompt e segmenta l'oggetto indicato.")
    print(
        "Nota: i frame normali del dominio servono ad AnomalyDINO, "
        "non all'inferenza zero-shot di AnomalyVFM."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
