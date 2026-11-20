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

"""Unit tests for the HH-RLHF formatters in `e2e_hh_rlhf_valuation`."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STUBBED = ("vllm", "datasets", "hnswlib", "torch", "sentence_transformers", "litellm")
with mock.patch.dict("sys.modules", {name: mock.MagicMock() for name in _STUBBED}):
    import e2e_hh_rlhf_valuation as hh

_ROW = {
    "chosen": "\n\nHuman: Hi\n\nAssistant: Hello! How can I help?",
    "rejected": "\n\nHuman: Hi\n\nAssistant: Go away.",
}


class HhRlhfFormatterTest(unittest.TestCase):

    def test_label_pair_matches_observed_label(self):
        observed = hh.hh_rlhf_response_formatter(_ROW)
        pair = json.loads(hh.hh_rlhf_label_pair_formatter(_ROW))
        self.assertEqual(pair["chosen"], observed)
        self.assertEqual({pair["chosen"], pair["rejected"]}, {" 1", " 2"})

    def test_observed_label_points_at_chosen_text(self):
        query = hh.hh_rlhf_query_formatter(_ROW)
        label = hh.hh_rlhf_response_formatter(_ROW).strip()
        option = query.split(f"Option {label}:\n")[1].split("\n")[0]
        self.assertEqual(option, "Hello! How can I help?")


if __name__ == "__main__":
    unittest.main()
