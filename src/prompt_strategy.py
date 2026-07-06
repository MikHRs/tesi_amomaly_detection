"""Legge e mostra le famiglie di prompt zero-shot."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FAMILIES = (
    "direct_anomaly_prompts",
    "background_prompts",
    "combined_prompts",
)


def load_prompts(config_path: Path) -> dict[str, list[str]]:
    """Carica e valida la configurazione YAML."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configurazione non trovata: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Il file YAML deve contenere una mappa di famiglie")

    prompts: dict[str, list[str]] = {}
    for family in FAMILIES:
        values = data.get(family, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"La famiglia {family} deve essere una lista di stringhe")
        prompts[family] = [value.strip() for value in values]
    return prompts


def print_prompts(prompts: dict[str, list[str]]) -> None:
    for family in FAMILIES:
        print(f"\n{family} ({len(prompts[family])})")
        for index, prompt in enumerate(prompts[family], start=1):
            print(f"  {index:02d}. {prompt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mostra le famiglie di prompt.")
    parser.add_argument("--config", type=Path, required=True, help="File YAML")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prompts = load_prompts(args.config)
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERRORE: {exc}")
        return 2
    print_prompts(prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
