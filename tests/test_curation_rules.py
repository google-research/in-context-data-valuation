# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `curation_rules` (calibrated margin and stratification)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation_rules import (
    CalibratedMarginRule,
    MetricSummary,
    UniquenessGuard,
    curate_graph,
    get_filtering_rule,
    registered_filtering_rules,
)
from influence_graph import QueryInfluenceGraph, _QueryNode


def _make_graph_with_nodes(nodes):
    g = QueryInfluenceGraph(enable_partial_update=False)
    for node in nodes:
        g.nodes.setdefault(node.query, {})[node.response] = node
    return g


class CalibratedMarginRuleUnitTest(unittest.TestCase):

    def test_registration_and_defaults(self):
        self.assertIn("calibrated_margin", registered_filtering_rules())
        rule = get_filtering_rule("calibrated_margin")
        self.assertIsInstance(rule, CalibratedMarginRule)
        self.assertTrue(rule.stratify_by_response)
        self.assertEqual(
            rule.uniqueness_guard,
            UniquenessGuard.ONE_OF_A_KIND_POSITIVE_INFLUENCE,
        )
        self.assertEqual(rule.outlier_sigmas, 2.0)
        self.assertEqual(rule.exemplified_ratio_weight, 0.5)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            CalibratedMarginRule(outlier_sigmas=0.0)
        with self.assertRaises(ValueError):
            CalibratedMarginRule(exemplified_ratio_weight=-0.1)

    def test_one_sided_exclusion_and_missing_ratio_imputation(self):
        rule = CalibratedMarginRule(
            outlier_sigmas=2.0, exemplified_ratio_weight=0.5
        )
        stats = {
            "base_score": MetricSummary(count=100, mean=-2.0, median=-2.0, std=1.0),
            "exemplified_score_ratio": MetricSummary(
                count=100, mean=0.5, median=0.5, std=0.4
            ),
        }
        scale = math.hypot(1.0, 0.5 * 0.4)

        # Upper-tail clean prototype: high positive z -> never excluded.
        clean = _QueryNode(
            query="q_clean", response="Yes", base_score=2.0, exemplified_score_ratio=1.5
        )
        self.assertFalse(rule.is_excluded(clean, stats))
        self.assertAlmostEqual(
            rule.rank_score(clean, stats), (4.0 + 0.5 * 1.0) / scale
        )

        # Lower-tail corrupted record: strongly negative z -> excluded.
        bad = _QueryNode(
            query="q_bad", response="Yes", base_score=-5.0, exemplified_score_ratio=-0.5
        )
        self.assertTrue(rule.is_excluded(bad, stats))
        self.assertLess(rule.rank_score(bad, stats), -2.0)

        # Isolated node with missing ratio imputes stratum mean (ratio delta = 0).
        isolated = _QueryNode(
            query="q_iso", response="Yes", base_score=-2.0, exemplified_score_ratio=None
        )
        self.assertFalse(rule.is_excluded(isolated, stats))
        self.assertAlmostEqual(rule.rank_score(isolated, stats), 0.0)

    def test_positive_influence_uniqueness_guard(self):
        rule = get_filtering_rule("calibrated_margin")
        good_unique = _QueryNode(
            query="q1",
            response="Yes",
            one_of_a_kind_ratio=1.0,
            exemplifying_advantage=0.5,
        )
        anti_hub_unique = _QueryNode(
            query="q2",
            response="Yes",
            one_of_a_kind_ratio=1.0,
            exemplifying_advantage=-0.3,
        )
        self.assertTrue(rule.is_unique(good_unique, 0.9))
        self.assertFalse(rule.is_unique(anti_hub_unique, 0.9))


class StratificationAndCurationTest(unittest.TestCase):

    def test_calibrated_margin_preserves_minority_class_under_prior_bias(self):
        """Response stratification prevents a lower-prior class from being starved."""
        nodes = []
        # Majority class "No" (10 nodes) has higher raw zero-shot scores (-1.0..-0.1).
        for i in range(10):
            nodes.append(
                _QueryNode(
                    query=f"q_no_{i}",
                    query_index=i,
                    response="No",
                    base_score=-1.0 + 0.1 * i,
                    exemplified_score_ratio=0.2,
                    one_of_a_kind_ratio=0.0,
                    exemplifying_advantage=0.1,
                )
            )
        # Minority class "Yes" (10 nodes) has much lower raw scores (-10.0..-9.1).
        for i in range(10):
            nodes.append(
                _QueryNode(
                    query=f"q_yes_{i}",
                    query_index=10 + i,
                    response="Yes",
                    base_score=-10.0 + 0.1 * i,
                    exemplified_score_ratio=0.2,
                    one_of_a_kind_ratio=0.0,
                    exemplifying_advantage=0.1,
                )
            )
        graph = _make_graph_with_nodes(nodes)
        res = curate_graph(graph, rule="calibrated_margin", target_percentage=0.5)
        self.assertTrue(res.stats.active)
        kept_responses = [n.response for n in res.kept_nodes]
        self.assertEqual(kept_responses.count("No"), 5)
        self.assertEqual(kept_responses.count("Yes"), 5)

    def test_open_domain_falls_back_to_global_stats(self):
        """Distinct long-form responses fall back cleanly to global ranking."""
        nodes = [
            _QueryNode(
                query=f"q_{i}",
                query_index=i,
                response=f"Unique long-form answer number {i}",
                base_score=-10.0 + i,
                exemplified_score_ratio=0.1 * i,
                one_of_a_kind_ratio=0.0,
                exemplifying_advantage=0.1,
            )
            for i in range(10)
        ]
        graph = _make_graph_with_nodes(nodes)
        res = curate_graph(graph, rule="calibrated_margin", target_percentage=0.5)
        self.assertFalse(res.stats.active)
        self.assertEqual(len(res.kept_nodes), 5)
        kept_indices = sorted(n.query_index for n in res.kept_nodes)
        self.assertEqual(kept_indices, [5, 6, 7, 8, 9])

    def test_anti_hub_unique_is_not_spared_from_lower_tail_exclusion(self):
        """A negative-margin anti-hub (adv <= 0) is excluded even if OOK = 1.0."""
        nodes = [
            _QueryNode(
                query=f"q_{i}",
                query_index=i,
                response="Yes" if i < 15 else "No",
                base_score=0.0,
                exemplified_score_ratio=0.0,
                one_of_a_kind_ratio=0.0,
                exemplifying_advantage=0.2,
            )
            for i in range(30)
        ]
        # Inject one severe lower-tail outlier with OOK=1.0 and adv=-1.0.
        nodes[0].base_score = -20.0
        nodes[0].one_of_a_kind_ratio = 1.0
        nodes[0].exemplifying_advantage = -1.0

        # And one genuine unique node (OOK=1.0, adv=+1.0) that IS spared.
        nodes[1].base_score = -20.0
        nodes[1].one_of_a_kind_ratio = 1.0
        nodes[1].exemplifying_advantage = 1.0

        graph = _make_graph_with_nodes(nodes)
        res = curate_graph(graph, rule="calibrated_margin", target_percentage=1.0)
        self.assertIn(("q_0", "Yes"), res.rule_excluded_keys)
        self.assertIn(("q_1", "Yes"), res.unique_kept_keys)
        self.assertNotIn(("q_1", "Yes"), res.rule_excluded_keys)


if __name__ == "__main__":
    unittest.main()
