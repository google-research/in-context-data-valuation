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

"""Unit tests for `influence_graph` (graph build, length norm, exemplifying advantage)."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from influence_graph import (
    QueryInfluenceGraph,
    build_query_graph_from_records,
)

_Q = ["Prompt: alpha\nResponse:", "Prompt: beta\nResponse:"]
_R = [" short answer here", " " + "word " * 39 + "end"]
_TOK = [4, 40]
_BASE = [-10.0, -100.0]
_EDGE = {(1, 0): -8.0, (0, 1): -92.0}  # 0 taught by 1: +2; 1 taught by 0: +8


def _make_two_node_records():
    recs = []
    for j in range(2):
        recs.append(
            {
                "prompt": _Q[j],
                "scoring_completions": _R[j],
                "query_text": _Q[j],
                "query_index": j,
                "example_query": None,
                "example_response": None,
                "is_base": True,
                "neighbor_index": None,
                "logprob": _BASE[j],
                "completion_num_tokens": _TOK[j],
            }
        )
        i = 1 - j
        recs.append(
            {
                "prompt": (
                    f"Answer ... <question>{_Q[i]}</question>"
                    f"<answer>{_R[i]}</answer>{_Q[j]}"
                ),
                "scoring_completions": _R[j],
                "query_text": _Q[j],
                "query_index": j,
                "example_query": _Q[i],
                "example_response": _R[i],
                "is_base": False,
                "neighbor_index": i,
                "logprob": _EDGE[(i, j)],
                "completion_num_tokens": _TOK[j],
            }
        )
    return recs


class InfluenceGraphBuildTest(unittest.TestCase):

    def test_metadata_build_and_length_normalization(self):
        recs = _make_two_node_records()

        g_none, errs_none = build_query_graph_from_records(
            recs, length_normalization="none"
        )
        self.assertEqual(errs_none, [])
        self.assertAlmostEqual(
            g_none.get_exemplifying_score_ratio(_Q[0], _R[0].strip()), 8.0
        )
        self.assertAlmostEqual(
            g_none.get_exemplifying_score_ratio(_Q[1], _R[1].strip()), 2.0
        )

        g_tok, errs_tok = build_query_graph_from_records(
            recs, length_normalization="tokens"
        )
        self.assertEqual(errs_tok, [])
        # Teacher 0 teaches node 1 (40 tokens): +8 / 40 = 0.2
        # Teacher 1 teaches node 0 (4 tokens): +2 / 4 = 0.5
        self.assertAlmostEqual(
            g_tok.get_exemplifying_score_ratio(_Q[0], _R[0].strip()), 0.2
        )
        self.assertAlmostEqual(
            g_tok.get_exemplifying_score_ratio(_Q[1], _R[1].strip()), 0.5
        )

    def test_save_and_reload_with_recomputed_normalization(self):
        recs = _make_two_node_records()
        g_none, _ = build_query_graph_from_records(
            recs, length_normalization="none"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.jsonl")
            g_none.save_to_jsonl(path)
            g_reloaded = QueryInfluenceGraph.load_from_jsonl(
                path, length_normalization="tokens"
            )
            self.assertAlmostEqual(
                g_reloaded.get_exemplifying_score_ratio(_Q[0], _R[0].strip()), 0.2
            )
            self.assertAlmostEqual(
                g_reloaded.get_exemplifying_score_ratio(_Q[1], _R[1].strip()), 0.5
            )


class LeaveQueryOutAdvantageTest(unittest.TestCase):

    def test_loo_excludes_target_edge_from_neighbor_reference(self):
        """Target node Q0's huge drop must not contaminate N1/N2's outgoing means."""
        g = QueryInfluenceGraph()
        for idx, q in enumerate(["Q0", "N1", "N2", "Other"]):
            g.add_base_query(q, query_index=idx, response=f"R{idx}", score=0.0)

        # Q0 teaches Other with delta = +5.0 -> Q0's own exemplifying ratio = 5.0
        g.add_exemplification("Other", "R3", "Q0", "R0", score=5.0)

        # N1 teaches Q0 with extreme delta = -100.0, and teaches Other with +2.0.
        # For Q0, N1's reference ratio MUST be +2.0 (excluding its edge to Q0).
        g.add_exemplification("Q0", "R0", "N1", "R1", score=-100.0)
        g.add_exemplification("Other", "R3", "N1", "R1", score=2.0)

        # N2 teaches Q0 with extreme delta = -200.0, and teaches Other with +8.0.
        # For Q0, N2's reference ratio MUST be +8.0 (excluding its edge to Q0).
        g.add_exemplification("Q0", "R0", "N2", "R2", score=-200.0)
        g.add_exemplification("Other", "R3", "N2", "R2", score=8.0)

        # Reference sample for Q0 is [2.0, 8.0, 5.0] -> mean=5.0, std=3.0, adv=0.0
        adv = g.get_exemplifying_advantage("Q0", "R0")
        self.assertIsNotNone(adv)
        self.assertAlmostEqual(adv, 0.0, places=6)

        node_q0 = g._get_node("Q0", "R0")
        self.assertAlmostEqual(node_q0.exemplifying_mean, 5.0, places=6)
        self.assertAlmostEqual(
            node_q0.exemplifying_std, float(np.std([2.0, 8.0, 5.0], ddof=1)), places=6
        )

    def test_loo_omits_neighbors_with_no_other_outgoing_edges(self):
        """A neighbor whose only outgoing edge targets Q0 is omitted (rem=0)."""
        g = QueryInfluenceGraph()
        g.add_base_query("Q0", query_index=0, response="R0", score=0.0)
        g.add_base_query("N1", query_index=1, response="R1", score=0.0)
        g.add_exemplification("Q0", "R0", "N1", "R1", score=1.0)
        g.add_exemplification("N1", "R1", "Q0", "R0", score=2.0)

        # N1 only teaches Q0, so excluding Q0 leaves 0 edges for N1.
        self.assertIsNone(g.get_exemplifying_advantage("Q0", "R0"))



class TruncatedPromptTest(unittest.TestCase):

    def test_truncated_one_shot_record_is_not_an_edge(self):
        recs = _make_two_node_records()
        for r in recs:
            if not r["is_base"] and r["query_index"] == 0:
                r["prompt_truncated"] = True
        graph, errors = build_query_graph_from_records(recs)
        self.assertEqual(len(errors), 1)
        self.assertIn("truncated", errors[0]["error"])
        node0 = graph._get_node(_Q[0].strip(), _R[0].strip())
        self.assertEqual(node0.exemplified_scores, [])


if __name__ == "__main__":
    unittest.main()
