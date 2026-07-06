import csv
import tempfile
import unittest
from pathlib import Path

from src.build_experiment_table import COLUMNS, build_experiment_table


class ExperimentTableTests(unittest.TestCase):
    def test_template_contains_only_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tables" / "experiments.csv"
            build_experiment_table(output)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [COLUMNS])

    def test_existing_table_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiments.csv"
            build_experiment_table(output)
            with self.assertRaises(FileExistsError):
                build_experiment_table(output)


if __name__ == "__main__":
    unittest.main()
