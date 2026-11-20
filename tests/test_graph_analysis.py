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

"""Unit tests for arbitration prompts and the arbitration cache in `graph_analysis`."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LITELLM = mock.MagicMock()
with mock.patch.dict("sys.modules", {"litellm": _LITELLM}):
    import graph_analysis

_PAIR = {
    "bad_query": "Q-high",
    "bad_response": "R-high",
    "bad_query_index": 3,
    "bad_score": 4,
    "neighbor_query": "Q-low",
    "neighbor_response": "R-low",
    "neighbor_query_index": 7,
    "neighbor_score": 1,
}


def _fake_completion(content):
    msg = mock.MagicMock()
    msg.content = content
    choice = mock.MagicMock()
    choice.message = msg
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


class BadNeighborPromptTest(unittest.TestCase):

    def test_record_names_and_scores(self):
        (prompt,) = graph_analysis.generate_llm_prompts_for_bad_neighbors(
            [_PAIR],
            data_review_instructions=["Own schema."],
            structured_output=False,
            record_names=("[Record A]", "[Record B]"),
            score_name="Human Helpfulness Score",
        )
        self.assertIn("[Record A]:\nInput: Q-high", prompt["user_prompt"])
        self.assertIn("Human Helpfulness Score: 4", prompt["user_prompt"])
        self.assertIn("[Record B]:\nInput: Q-low", prompt["user_prompt"])
        self.assertIn("Human Helpfulness Score: 1", prompt["user_prompt"])
        self.assertNotIn("JSON format", prompt["system_instruction"])


class ArbitrationCacheTest(unittest.TestCase):

    def test_cache_is_per_model_and_retries_bad_json(self):
        prompts = graph_analysis.generate_llm_prompts_for_bad_neighbors([_PAIR])
        verdict = json.dumps({"is_contradiction": False, "verdict": "Neither"})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "evals.jsonl")
            _LITELLM.completion.side_effect = [
                _fake_completion("not json"),
                _fake_completion(verdict),
            ]
            (res,) = graph_analysis.evaluate_contradictions_with_llm(
                prompts, model_name="provider/model-a", output_path=path
            )
            self.assertEqual(res["llm_evaluation"]["verdict"], "Neither")
            self.assertEqual(_LITELLM.completion.call_count, 2)
            self.assertTrue(_LITELLM.completion.call_args.kwargs["drop_params"])
            self.assertNotIn("api_key", _LITELLM.completion.call_args.kwargs)

            # Same model: served from the cache.
            _LITELLM.completion.reset_mock(side_effect=True)
            graph_analysis.evaluate_contradictions_with_llm(
                prompts, model_name="provider/model-a", output_path=path
            )
            _LITELLM.completion.assert_not_called()

            # Different model: evaluated again.
            _LITELLM.completion.side_effect = [_fake_completion(verdict)]
            graph_analysis.evaluate_contradictions_with_llm(
                prompts, model_name="provider/model-b", output_path=path
            )
            self.assertEqual(_LITELLM.completion.call_count, 1)


class RejectedParameterTest(unittest.TestCase):

    def test_retries_without_rejected_temperature(self):
        prompts = graph_analysis.generate_llm_prompts_for_bad_neighbors([_PAIR])
        verdict = json.dumps({"is_contradiction": False, "verdict": "Neither"})
        _LITELLM.completion.reset_mock(side_effect=True)
        _LITELLM.completion.side_effect = [
            Exception("BadRequestError: `temperature` is deprecated for this model."),
            _fake_completion(verdict),
        ]
        (res,) = graph_analysis.evaluate_contradictions_with_llm(
            prompts, model_name="provider/new-model", temperature=0
        )
        self.assertEqual(res["llm_evaluation"]["verdict"], "Neither")
        first, second = _LITELLM.completion.call_args_list
        self.assertIn("temperature", first.kwargs)
        self.assertNotIn("temperature", second.kwargs)
        self.assertIn("response_format", second.kwargs)


def _small_graph():
    """4 records (two answer labels) with all-pairs 1-shot edges and varied scores."""
    from influence_graph import build_query_graph_from_records

    queries = [f"Q{i}" for i in range(4)]
    labels = [" 1", " 2", " 1", " 2"]
    base = [-0.2, -0.5, -1.5, -0.3]
    shift = {0: 0.1, 1: -0.4, 2: 0.3, 3: -0.1}  # mean shift each record receives
    recs = []
    for i in range(4):
        recs.append({"query_text": queries[i], "query_index": i, "scoring_completions": labels[i],
                     "example_query": None, "example_response": None, "logprob": base[i],
                     "completion_num_tokens": 1})
        for j in range(4):
            if j != i:
                recs.append({"query_text": queries[i], "query_index": i,
                             "scoring_completions": labels[i], "example_query": queries[j],
                             "example_response": labels[j],
                             "logprob": base[i] + shift[i] + 0.01 * j,
                             "completion_num_tokens": 1})
    graph, errors = build_query_graph_from_records(recs, errors_output_path=os.devnull)
    assert not errors, errors
    return graph


class RuleAnchorTest(unittest.TestCase):

    def test_rank_records_by_rule_is_ascending_rule_score(self):
        from curation_rules import curate_graph

        graph = _small_graph()
        expected = curate_graph(graph, rule="calibrated_margin").rank_scores
        ranked = graph_analysis.rank_records_by_rule(graph, "calibrated_margin")
        self.assertEqual(len(ranked), 4)
        scores = [r["rule_score"] for r in ranked]
        self.assertEqual(scores, sorted(scores))
        for r in ranked:
            self.assertAlmostEqual(r["rule_score"], expected[(r["query"], r["response"])])
        self.assertEqual(len(graph_analysis.rank_records_by_rule(graph, "calibrated_margin", budget=2)), 2)

    def test_exemplifying_advantage_rule_ranks_by_adv_delta(self):
        graph = _small_graph()
        ranked = graph_analysis.rank_records_by_rule(graph, "exemplifying_advantage")
        advs = [r["exemplifying_advantage"] for r in ranked]
        self.assertEqual(advs, sorted(advs))

    def test_worst_neighbor_twins_with_rule_uses_rule_order(self):
        graph = _small_graph()
        ranked = graph_analysis.rank_records_by_rule(graph, "calibrated_margin", budget=2)
        pairs = graph_analysis.find_worst_neighbor_twins(graph, budget=2, anchor_rule="calibrated_margin")
        anchors = {(q, p["query_response"]) for q, ps in pairs.items() for p in ps}
        self.assertEqual(anchors, {(r["query"], r["response"]) for r in ranked})
        for ps in pairs.values():
            for p in ps:
                self.assertIn("rule_score", p)
                self.assertNotEqual(p["example_query"], p["query_response"])

    def test_worst_neighbor_twins_paper_mode_unchanged(self):
        graph = _small_graph()
        pairs = graph_analysis.find_worst_neighbor_twins(graph, budget=4)
        devs = sorted(p["abs_adv_e_deviation"] for ps in pairs.values() for p in ps)
        self.assertEqual(len(devs), 4)
        for ps in pairs.values():
            for p in ps:
                self.assertNotIn("rule_score", p)


if __name__ == "__main__":
    unittest.main()
