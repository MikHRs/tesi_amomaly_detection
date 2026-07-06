import unittest

from src.anomaly_output_to_box import validate_parameters


class AnomalyOutputToBoxTests(unittest.TestCase):
    def test_parameters(self) -> None:
        validate_parameters(0.5, 50)
        with self.assertRaises(ValueError):
            validate_parameters(1.1, 50)
        with self.assertRaises(ValueError):
            validate_parameters(0.5, 0)


if __name__ == "__main__":
    unittest.main()
