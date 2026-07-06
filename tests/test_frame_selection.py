import unittest

from src.extract_normal_frames import evenly_spaced_frame_indices
from src.extract_test_anomaly_frames import (
    parse_comma_separated,
    selected_frame_indices,
)


class NormalFrameSelectionTests(unittest.TestCase):
    def test_evenly_spaced_indices_are_not_consecutive(self) -> None:
        indices = evenly_spaced_frame_indices(frame_count=1000, num_frames=200)
        self.assertEqual(len(indices), 200)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 999)
        self.assertTrue(
            all(right - left > 1 for left, right in zip(indices, indices[1:]))
        )

    def test_rejects_too_many_non_consecutive_frames(self) -> None:
        with self.assertRaises(ValueError):
            evenly_spaced_frame_indices(frame_count=10, num_frames=6)


class AnomalyFrameSelectionTests(unittest.TestCase):
    def test_times_are_converted_and_deduplicated(self) -> None:
        indices = selected_frame_indices(
            frame_count=500,
            source_fps=25.0,
            times=[2.0, 1.0, 2.0],
        )
        self.assertEqual(indices, [25, 50])

    def test_frame_indices_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            selected_frame_indices(
                frame_count=100,
                source_fps=25.0,
                frames=[100],
            )

    def test_comma_separated_parser(self) -> None:
        self.assertEqual(parse_comma_separated("1, 2,3", int), [1, 2, 3])
        with self.assertRaises(ValueError):
            parse_comma_separated("1,,3", int)


if __name__ == "__main__":
    unittest.main()
