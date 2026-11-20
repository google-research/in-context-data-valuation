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

"""Unit tests for neighbor selection and prompt generation in `scoring`."""

import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STUBBED = (
    "vllm",
    "datasets",
    "hnswlib",
    "torch",
    "sentence_transformers",
)
with mock.patch.dict(
    "sys.modules", {name: mock.MagicMock() for name in _STUBBED}
):
    from scoring import (
        _load_scored_prefix,
        _select_neighbor_indices,
        generate_dataset_inputs,
        score_inputs,
    )


class SelectNeighborIndicesTest(unittest.TestCase):

    def test_excludes_query_when_first(self):
        self.assertEqual(
            _select_neighbor_indices([5, 1, 2, 3], query_index=5, max_neighbors=3),
            [1, 2, 3],
        )

    def test_excludes_query_when_not_first(self):
        self.assertEqual(
            _select_neighbor_indices([1, 5, 2, 3], query_index=5, max_neighbors=3),
            [1, 2, 3],
        )

    def test_keeps_extra_neighbor_when_query_absent(self):
        self.assertEqual(
            _select_neighbor_indices([1, 2, 3], query_index=5, max_neighbors=3),
            [1, 2, 3],
        )

    def test_respects_max_neighbors(self):
        self.assertEqual(
            _select_neighbor_indices([1, 2, 3, 4, 5], query_index=9, max_neighbors=2),
            [1, 2],
        )

    def test_preserves_distance_order(self):
        self.assertEqual(
            _select_neighbor_indices([7, 4, 9, 4, 1], query_index=9, max_neighbors=4),
            [7, 4, 4, 1],
        )

    def test_accepts_numpy_values(self):
        row = np.array([3, 5, 1], dtype=np.uint64)
        got = _select_neighbor_indices(row, query_index=5, max_neighbors=2)
        self.assertEqual(got, [3, 1])
        self.assertTrue(all(isinstance(v, int) for v in got))


class _FakeDataset:
    """Minimal HuggingFace Dataset-like wrapper with row-count `__len__`."""

    def __init__(self, columns):
        self._columns = columns

    def __getitem__(self, key):
        return self._columns[key]

    def __len__(self):
        return len(next(iter(self._columns.values())))


class GenerateDatasetInputsTest(unittest.TestCase):

    def setUp(self):
        self.dataset = _FakeDataset(
            {
                "text": ["Prompt A", "Prompt A", "Prompt B", "Prompt B"],
                "answer": ["Resp A0", "Resp A1", "Resp B0", "Resp B1"],
            }
        )
        # Exact-duplicate prompts can place the query at position > 0.
        self.labels = np.array(
            [
                [1, 0, 2, 3],
                [0, 1, 2, 3],
                [3, 2, 0, 1],
                [2, 3, 0, 1],
            ],
            dtype=np.uint64,
        )

    def test_never_uses_self_as_exemplar(self):
        records = generate_dataset_inputs(
            self.dataset, self.labels, text_col="text", answer_col="answer"
        )
        # 4 queries * (1 base + 3 neighbors) = 16 records
        self.assertEqual(len(records), 16)
        for rec in records:
            if rec["is_base"]:
                self.assertIsNone(rec["neighbor_index"])
            else:
                self.assertNotEqual(rec["query_index"], rec["neighbor_index"])

    def test_score_difference_mode_never_uses_self(self):
        diff_dataset = _FakeDataset(
            {
                "text": ["Prompt A", "Prompt A", "Prompt B", "Prompt B"],
                "answer": [
                    json.dumps({"chosen": f"C{i}", "rejected": f"R{i}"})
                    for i in range(4)
                ],
            }
        )
        records = generate_dataset_inputs(
            diff_dataset,
            self.labels,
            text_col="text",
            answer_col="answer",
            score_difference=True,
        )
        # 4 queries * (1 base + 3 neighbors) * 2 (chosen + rejected) = 32
        self.assertEqual(len(records), 32)
        for rec in records:
            if not rec["is_base"]:
                self.assertNotEqual(rec["query_index"], rec["neighbor_index"])
                # Neighbors are always shown with their chosen response.
                self.assertEqual(rec["example_response"], f"C{rec['neighbor_index']}")
            expected = f"C{rec['query_index']}" if rec["is_chosen"] else f"R{rec['query_index']}"
            self.assertEqual(rec["scoring_completions"], expected)

    def test_prompt_format(self):
        records = generate_dataset_inputs(
            self.dataset, self.labels, text_col="text", answer_col="answer"
        )
        base, one_shot = records[0], records[1]
        self.assertEqual(base["prompt"], "Prompt A")
        self.assertEqual(
            one_shot["prompt"],
            "Answer the question considering the example question and answer"
            " pair. <question>Prompt A</question><answer>Resp A1</answer>Prompt A",
        )
        # The dataset row index is metadata only, never part of the prompt.
        for rec in records:
            self.assertNotIn(f"<question>{rec['query_index']}</question>", rec["prompt"])


class ScoreInputsResumeTest(unittest.TestCase):

    def _records(self, n):
        return [
            {"query_index": i, "is_base": True, "neighbor_index": None,
             "prompt": f"P{i}", "scoring_completions": " 1"}
            for i in range(n)
        ]

    def test_drops_incomplete_trailing_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scored.jsonl")
            done = self._records(2)
            with open(path, "w") as f:
                f.write(json.dumps(done[0]) + "\n" + json.dumps(done[1]) + "\n")
                f.write('{"query_index": 2, "is_ba')  # interrupted write
            self.assertEqual(_load_scored_prefix(path), done)
            with open(path) as f:
                self.assertEqual(len(f.read().splitlines()), 2)

    def test_resume_rejects_stale_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scored.jsonl")
            stale = self._records(1)
            stale[0]["prompt"] = "old prompt format"
            with open(path, "w") as f:
                f.write(json.dumps(stale[0]) + "\n")
            with self.assertRaises(ValueError):
                score_inputs(None, None, self._records(2), output_path=path)

    def test_resume_returns_finished_records_without_scoring(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scored.jsonl")
            done = self._records(2)
            with open(path, "w") as f:
                for r in done:
                    f.write(json.dumps(dict(r, logprob=-1.0)) + "\n")
            def _no_engine():
                raise AssertionError("the model must not be loaded")

            out = score_inputs(
                None, None, self._records(2), output_path=path, llm_factory=_no_engine
            )
            self.assertEqual([r["logprob"] for r in out], [-1.0, -1.0])

    def test_needs_llm_or_factory_when_records_remain(self):
        with self.assertRaises(ValueError):
            score_inputs(None, None, self._records(1))


if __name__ == "__main__":
    unittest.main()
