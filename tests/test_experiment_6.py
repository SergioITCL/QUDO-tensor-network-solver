import unittest

from experimentation.experiment_6_scip_time_to_tn.experiment_6_scip_time_to_tn import (
    summarize_results,
)


class Experiment6SummaryTests(unittest.TestCase):
    def test_censored_instances_are_excluded_from_target_statistics(self):
        results = [
            {
                "target_reached": True,
                "matrix_time": 0.5,
                "time_to_target": 2.0,
                "time_ratio": 4.0,
            },
            {
                "target_reached": False,
                "matrix_time": 1.0,
                "time_to_target": None,
                "time_ratio": None,
                "ratio_lower_bound": 60.0,
            },
            {
                "target_reached": True,
                "matrix_time": 0.25,
                "time_to_target": 3.0,
                "time_ratio": 12.0,
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["n_instances"], 3)
        self.assertEqual(summary["n_target_reached"], 2)
        self.assertEqual(summary["n_target_not_reached"], 1)
        self.assertAlmostEqual(summary["target_success_rate"], 2 / 3)
        self.assertEqual(summary["scip_time_to_target"]["mean"], 2.5)
        self.assertEqual(summary["scip_time_to_target"]["median"], 2.5)
        self.assertEqual(summary["time_ratio"]["mean"], 8.0)
        self.assertEqual(summary["time_ratio"]["min"], 4.0)
        self.assertEqual(summary["time_ratio"]["max"], 12.0)

    def test_all_censored_configuration_has_no_fake_timeout_mean(self):
        summary = summarize_results(
            [
                {
                    "target_reached": False,
                    "matrix_time": 0.5,
                    "time_to_target": None,
                    "time_ratio": None,
                }
            ]
        )

        self.assertIsNone(summary["scip_time_to_target"]["mean"])
        self.assertIsNone(summary["time_ratio"]["median"])


if __name__ == "__main__":
    unittest.main()
