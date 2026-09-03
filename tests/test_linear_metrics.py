import inspect
import unittest
from unittest import mock

import numpy as np

import libmultilabel.linear.metrics as metrics_module
from libmultilabel.linear.metrics import (
    PropensityScoredPrecisionAtK,
    compute_metrics,
    get_metrics,
    tabulate_metrics,
)


class PropensityScoredPrecisionAtKTest(unittest.TestCase):
    def setUp(self):
        self.label_pos_counts = np.array([0, 1, 3, 7])
        self.num_instances = 10
        self.propensity_a = 1.0
        self.propensity_b = 1.0
        self.preds = np.array(
            [
                [4.0, 3.0, 2.0, 1.0],
                [2.0, 1.0, 4.0, 3.0],
            ]
        )
        self.target = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        )

    def make_metric(self, top_k=2):
        return PropensityScoredPrecisionAtK(
            top_k=top_k,
            label_pos_counts=self.label_pos_counts,
            N=self.num_instances,
            A=self.propensity_a,
            B=self.propensity_b,
        )

    def test_inverse_propensity_formula_and_factory_parameters(self):
        collection = get_metrics(
            ["PSP@2"],
            num_classes=4,
            label_pos_counts=self.label_pos_counts,
            num_instances=self.num_instances,
            propensity_a=self.propensity_a,
            propensity_b=self.propensity_b,
        )
        metric = collection.metrics["PSP@2"]

        C = (np.log(self.num_instances) - 1.0) * (self.propensity_b + 1.0) ** self.propensity_a
        expected = 1.0 + C * np.power(
            self.label_pos_counts.astype(np.float64) + self.propensity_b,
            -self.propensity_a,
        )
        np.testing.assert_allclose(metric.inv_propensity, expected)

        score = compute_metrics(
            self.preds,
            self.target,
            ["PSP@2"],
            label_pos_counts=self.label_pos_counts,
            num_instances=self.num_instances,
            propensity_a=self.propensity_a,
            propensity_b=self.propensity_b,
        )["PSP@2"]
        self.assertAlmostEqual(score, 0.5549787538484547)

    def test_normalized_psp_uses_dataset_wide_ratio_and_returns_fraction(self):
        metric = self.make_metric()
        metric.update(self.preds, self.target)

        expected = 0.5549787538484547
        per_row_normalization = 0.5256123134547396
        self.assertAlmostEqual(metric.compute(), expected)
        self.assertNotAlmostEqual(metric.compute(), per_row_normalization)
        self.assertIn("55.50", tabulate_metrics({"PSP@2": metric.compute()}, "test"))

    def test_updates_are_batch_invariant(self):
        full_batch_metric = self.make_metric()
        full_batch_metric.update(self.preds, self.target)

        split_batch_metric = self.make_metric()
        split_batch_metric.update(self.preds[:1], self.target[:1])
        split_batch_metric.update(self.preds[1:], self.target[1:])

        self.assertAlmostEqual(full_batch_metric.compute(), split_batch_metric.compute())

    def test_update_and_update_argsort_are_equivalent(self):
        direct_metric = self.make_metric()
        direct_metric.update(self.preds, self.target)

        shared_ranking_metric = self.make_metric()
        shared_ranking_metric.update_argsort(metrics_module._argsort_top_k(self.preds, 2), self.target)

        self.assertAlmostEqual(direct_metric.compute(), shared_ranking_metric.compute())

    def test_perfect_missed_and_empty_targets(self):
        target = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        metric = self.make_metric(top_k=1)
        metric.update(np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]), target)
        self.assertEqual(metric.compute(), 1.0)

        metric.reset()
        self.assertEqual(metric.compute(), 0.0)
        metric.update(np.array([[1.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0]]), target)
        self.assertEqual(metric.compute(), 0.0)

        metric.reset()
        metric.update(self.preds, np.zeros_like(self.target))
        self.assertEqual(metric.compute(), 0.0)

        metric = self.make_metric()
        metric.update(self.preds, self.target)
        expected = metric.compute()
        metric.reset()
        metric.update(
            np.vstack((self.preds, np.array([[4.0, 3.0, 2.0, 1.0]]))),
            np.vstack((self.target, np.zeros((1, 4), dtype=int))),
        )
        self.assertAlmostEqual(metric.compute(), expected)

    def test_metric_collection_reuses_prediction_partition(self):
        collection = get_metrics(
            ["P@1", "PSP@1", "PSP@2"],
            num_classes=4,
            label_pos_counts=self.label_pos_counts,
            num_instances=self.num_instances,
            propensity_a=self.propensity_a,
            propensity_b=self.propensity_b,
        )
        original_argpartition = np.argpartition

        with mock.patch.object(metrics_module.np, "argpartition", wraps=original_argpartition) as argpartition:
            collection.update(self.preds, self.target)

        self.assertEqual(argpartition.call_count, 1)

    def test_rejects_invalid_label_metadata(self):
        with self.assertRaisesRegex(ValueError, "label_pos_counts is required"):
            get_metrics(["PSP@1"], num_classes=4, num_instances=self.num_instances)
        with self.assertRaisesRegex(ValueError, r"len\(label_pos_counts\)"):
            get_metrics(
                ["PSP@1"],
                num_classes=4,
                label_pos_counts=np.array([0, 1, 2]),
                num_instances=self.num_instances,
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            PropensityScoredPrecisionAtK(1, np.array([0, -1]), self.num_instances)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            PropensityScoredPrecisionAtK(1, np.array([0, np.inf]), self.num_instances)
        with self.assertRaisesRegex(ValueError, "top_k"):
            PropensityScoredPrecisionAtK(3, np.array([0, 1]), self.num_instances)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PropensityScoredPrecisionAtK(1, np.array([0, 1]), 0)
        with self.assertRaisesRegex(ValueError, "target label dimension"):
            self.make_metric().update(self.preds[:, :3], self.target[:, :3])

    def test_rejects_nonbinary_target(self):
        target = self.target.astype(np.float64)
        target[0, 0] = 0.5

        with self.assertRaisesRegex(ValueError, "target must be binary"):
            self.make_metric().update(self.preds, target)

    def test_compute_metrics_preserves_positional_multiclass(self):
        preds = np.array([[0.9, 0.8], [0.8, 0.9]])
        target = np.array([[1, 0], [0, 1]])

        positional = compute_metrics(preds, target, ["Micro-F1"], True)["Micro-F1"]
        keyword = compute_metrics(preds, target, ["Micro-F1"], multiclass=True)["Micro-F1"]
        multilabel = compute_metrics(preds, target, ["Micro-F1"])["Micro-F1"]

        self.assertEqual(positional, 1.0)
        self.assertEqual(positional, keyword)
        self.assertAlmostEqual(multilabel, 2.0 / 3.0)

    def test_get_metrics_preserves_positional_multiclass(self):
        collection = get_metrics(["Micro-F1"], 2, True)

        self.assertTrue(collection.metrics["Micro-F1"].multiclass)

    def test_auxiliary_metric_metadata_is_keyword_only(self):
        for function in (get_metrics, compute_metrics):
            parameters = inspect.signature(function).parameters
            for name in (
                "unseen_labels",
                "label_pos_counts",
                "num_instances",
                "propensity_a",
                "propensity_b",
            ):
                with self.subTest(function=function.__name__, parameter=name):
                    self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)


if __name__ == "__main__":
    unittest.main()
