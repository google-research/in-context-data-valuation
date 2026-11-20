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

"""Unit tests for duplicate-row handling in `valuation_pipeline` and `scoring`."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STUBBED = ("vllm", "datasets", "hnswlib", "torch", "sentence_transformers")
with mock.patch.dict("sys.modules", {name: mock.MagicMock() for name in _STUBBED}):
    import valuation_pipeline
    from scoring import first_occurrence_rows, generate_dataset_inputs
    from influence_graph import build_query_graph_from_records


def _brute_force_neighbors(embeddings, query_embeddings, k):
    sims = query_embeddings @ embeddings.T
    labels = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return labels.astype(np.uint64), np.take_along_axis(1 - sims, labels, axis=1)


class _FakeDataset:
    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, key):
        return self._columns[key]

    def __len__(self):
        return len(next(iter(self._columns.values())))


# Rows 0 and 2 are exact duplicates; row 3 shares row 0's query with another label.
_TEXT = ["Q0", "Q1", "Q0", "Q0", "Q4"]
_ANSWER = [" 1", " 2", " 1", " 2", " 1"]
_EMB = np.array(
    [[1.0, 0.0], [0.6, 0.8], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
)


class DuplicateRowsTest(unittest.TestCase):

    def setUp(self):
        self.dataset = _FakeDataset(
            {
                "__formatted_query__": _TEXT,
                "__formatted_response__": _ANSWER,
                "__formatted_embedding__": _TEXT,
            }
        )
        self.pipeline = valuation_pipeline.DataValuationPipeline(
            vllm_config={},
            embedding_model_name="unused",
            k_neighbors=3,
            output_graph_path="unused.jsonl",
            text_col="text",
            answer_col="answer",
        )

    def test_first_occurrence_rows(self):
        self.assertEqual(first_occurrence_rows(_TEXT, _ANSWER), [0, 1, 3, 4])

    def test_duplicates_are_never_neighbors_or_queries(self):
        keep = first_occurrence_rows(_TEXT, _ANSWER)
        with mock.patch.object(
            valuation_pipeline, "get_embeddings", return_value=_EMB
        ), mock.patch.object(
            valuation_pipeline, "build_hnsw_index", side_effect=lambda e: e
        ), mock.patch.object(
            valuation_pipeline, "find_neighbors", side_effect=_brute_force_neighbors
        ):
            labels, _ = self.pipeline.find_neighbors(self.dataset, keep)

        self.assertEqual(labels.shape, (5, 3))
        self.assertTrue(np.all(labels[2] == -1))  # the duplicate row is not queried
        self.assertNotIn(2, labels)  # ... and never appears as a neighbor
        self.assertTrue(set(labels[keep].ravel()) <= set(keep))

        records = generate_dataset_inputs(
            self.dataset, labels, "__formatted_query__", "__formatted_response__",
            query_rows=keep,
        )
        self.assertEqual({r["query_index"] for r in records}, set(keep))
        for r in records:
            if not r["is_base"]:
                self.assertNotEqual(
                    (r["example_query"], r["example_response"]),
                    (r["query_text"], r["scoring_completions"]),
                )

        # The graph has one node per unique row and no self-loops.
        for r in records:
            r["logprob"] = -1.0
            r["completion_num_tokens"] = 1
        graph, errors = build_query_graph_from_records(
            records=records, errors_output_path=os.devnull
        )
        self.assertEqual(errors, [])
        nodes = [n for by_resp in graph.nodes.values() for n in by_resp.values()]
        self.assertEqual(len(nodes), len(keep))
        for node in nodes:
            for edge in node.exemplified_scores:
                self.assertNotEqual(
                    (edge.query, edge.response), (node.query, node.response)
                )


if __name__ == "__main__":
    unittest.main()
