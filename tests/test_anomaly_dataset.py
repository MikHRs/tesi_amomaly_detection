import unittest

from src.build_anomaly_dataset import (
    classify_sampled_frames,
    sample_frame_indices,
    split_normal_frames,
)


class AnomalyDatasetTests(unittest.TestCase):
    def test_sample_frame_indices_two_fps_from_twenty_five(self) -> None:
        indices = sample_frame_indices(
            frame_count=100,
            source_fps=25.0,
            sample_fps=2.0,
        )
        self.assertEqual(indices, [0, 12, 25, 38, 50, 62, 75, 88])

    def test_classification_excludes_transition_margin(self) -> None:
        sampled = [0, 10, 20, 30, 40, 50, 60]
        active = set(range(30, 61))

        normal, anomaly, ambiguous = classify_sampled_frames(
            sampled,
            active,
            frame_count=70,
            margin_frames=5,
        )

        self.assertEqual(normal, [0, 10, 20])
        self.assertEqual(anomaly, [40, 50])
        self.assertEqual(ambiguous, [30, 60])

    def test_split_normal_frames_is_chronological(self) -> None:
        train, validation = split_normal_frames([0, 10, 20, 30, 40], 0.8)
        self.assertEqual(train, [0, 10, 20, 30])
        self.assertEqual(validation, [40])


if __name__ == "__main__":
    unittest.main()
