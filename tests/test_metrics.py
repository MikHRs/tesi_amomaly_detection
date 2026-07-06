import unittest

from src.metrics import box_area, evaluate_frame, iou, match_boxes


class MetricsTests(unittest.TestCase):
    def test_box_area_and_iou(self) -> None:
        self.assertEqual(box_area([0, 0, 10, 10]), 100)
        self.assertEqual(box_area([10, 10, 0, 0]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_one_to_one_matching(self) -> None:
        predictions = [[0, 0, 10, 10], [0, 0, 9, 9]]
        ground_truth = [[0, 0, 10, 10]]
        self.assertEqual(match_boxes(predictions, ground_truth, 0.5), (1, 1, 0))

    def test_evaluate_frame(self) -> None:
        result = evaluate_frame([[0, 0, 10, 10]], [[0, 0, 10, 10]], 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
