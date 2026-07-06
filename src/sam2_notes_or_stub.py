"""Controllo leggero e promemoria per una futura prova con SAM2.

Pseudo-pipeline:

1. caricare video/frame;
2. scegliere il frame iniziale;
3. fornire un prompt punto, box o maschera;
4. segmentare l'oggetto candidato;
5. propagare la maschera nei frame successivi;
6. salvare maschere e/o box;
7. correggere manualmente errori e drift.

SAM2 non decide cosa sia anomalo; serve come supporto per segmentare/tracciare
oggetti candidati. Un eventuale prompt testuale richiede un detector o un
altro modulo che produca il prompt geometrico per SAM2.

Questo file non scarica checkpoint, non importa torch e non esegue inferenza.
"""

from __future__ import annotations

from importlib.util import find_spec


def module_available(name: str) -> bool:
    """Controlla un modulo senza importarlo."""
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    torch_available = module_available("torch")
    sam2_available = module_available("sam2")

    print("Controllo ambiente opzionale SAM2")
    print(f"- torch installato: {'sì' if torch_available else 'no'}")
    print(f"- modulo sam2 installato: {'sì' if sam2_available else 'no'}")
    print(
        "\nPromemoria: SAM2 non decide cosa sia anomalo; segmenta e segue "
        "un candidato indicato con un prompt."
    )

    if not (torch_available and sam2_available):
        print(
            "\nSAM2 non è pronto, ma la pipeline leggera continua a funzionare.\n"
            "Per una prova futura:\n"
            "1. creare un ambiente separato;\n"
            "2. seguire le istruzioni del repository ufficiale facebookresearch/sam2;\n"
            "3. installare una build di PyTorch adatta a CPU/GPU;\n"
            "4. scaricare manualmente un checkpoint piccolo;\n"
            "5. registrare versione, checkpoint e hardware nel diario.\n"
            "Non sono stati installati pacchetti o scaricati modelli."
        )
        return 0

    print(
        "\nDipendenze individuate. Questo script resta intenzionalmente uno stub: "
        "l'inferenza va aggiunta solo dopo avere fissato video, prompt e checkpoint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
