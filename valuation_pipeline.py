# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import numpy as np
import os
import gc
import torch
import time
from typing import List, Dict, Any, Optional, Callable
from datasets import load_dataset, Dataset


os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM

from neighbor_finding import get_embeddings, build_hnsw_index, find_neighbors
from scoring import first_occurrence_rows, generate_dataset_inputs, score_inputs
from influence_graph import build_query_graph_from_records, QueryInfluenceGraph

class DataValuationPipeline:
    """
    A pipeline for data valuation.
    Each step is broken down into a method that can be overridden for customization.
    """

    def __init__(
        self,
        vllm_config: Dict[str, Any],
        embedding_model_name: str,
        k_neighbors: int,
        output_graph_path: str,
        prompt_logprobs: int = 10,
        text_col: Optional[str] = None,
        answer_col: Optional[str] = None,
        query_formatting_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        response_formatting_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        embedding_formatting_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        embedding_batch_size: int = 32,
        errors_output_path: str = "errors.jsonl",
        output_embeddings_path: Optional[str] = None,
        output_neighbors_path: Optional[str] = None,
        output_scoring_inputs_path: Optional[str] = None,
        output_scored_records_path: Optional[str] = None,
        score_difference: bool = False,
        length_normalization: str = "none",
    ):
        """
        Args:
          k_neighbors: Size of the nearest-neighbor query, including the record
            itself (`k_neighbors - 1` one-shot exemplars are scored per record).
          length_normalization: Passed to the graph; "none" (total nats),
            "tokens" (per completion token), or "chars".
        """
        self.text_col = text_col
        self.answer_col = answer_col
        self.query_formatting_fn = query_formatting_fn
        self.response_formatting_fn = response_formatting_fn
        self.embedding_formatting_fn = embedding_formatting_fn
        self.vllm_config = vllm_config
        self.embedding_model_name = embedding_model_name
        self.prompt_logprobs = prompt_logprobs
        self.embedding_batch_size = embedding_batch_size
        self.k_neighbors = k_neighbors
        self.output_graph_path = output_graph_path
        self.errors_output_path = errors_output_path
        self.score_difference = score_difference
        self.length_normalization = length_normalization
        if score_difference and length_normalization != "none":
            raise ValueError(
                "length_normalization must be 'none' with score_difference=True: "
                "scores are chosen-minus-rejected differences, which have no single "
                "response length."
            )

        if output_embeddings_path:
            if not output_embeddings_path.endswith(".npy"):
                output_embeddings_path += ".npy"
            self.output_embeddings_path = os.path.abspath(output_embeddings_path)
        else:
            self.output_embeddings_path = None

        if output_neighbors_path:
            if not output_neighbors_path.endswith(".npz"):
                output_neighbors_path += ".npz"
            self.output_neighbors_path = os.path.abspath(output_neighbors_path)
        else:
            self.output_neighbors_path = None

        if output_scoring_inputs_path:
            self.output_scoring_inputs_path = os.path.abspath(
                output_scoring_inputs_path
            )
        else:
            self.output_scoring_inputs_path = None

        if output_scored_records_path:
            self.output_scored_records_path = os.path.abspath(
                output_scored_records_path
            )
        else:
            self.output_scored_records_path = None

        if not self.text_col and not self.query_formatting_fn:
            raise ValueError("Must provide either 'text_col' or 'query_formatting_fn'.")
        if not self.answer_col and not self.response_formatting_fn:
            raise ValueError(
                "Must provide either 'answer_col' or 'response_formatting_fn'."
            )

        self._internal_text_col = "__formatted_query__"
        self._internal_answer_col = "__formatted_response__"
        self._internal_embedding_col = "__formatted_embedding__"

    def load_data(self, dataset_name: str, dataset_split: str, **kwargs) -> Dataset:
        """Step 1: Loads the dataset. Can be overridden to load from local files."""
        print(f"Loading dataset: {dataset_name} split='{dataset_split}'")
        return load_dataset(dataset_name, split=dataset_split, **kwargs)

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Step 1.5: Formats the dataset using custom functions or default columns."""
        print("Formatting dataset records...")

        def _apply_format(row):
            query_text = (
                self.query_formatting_fn(row)
                if self.query_formatting_fn
                else str(row[self.text_col])
            )
            return {
                self._internal_text_col: query_text,
                self._internal_answer_col: self.response_formatting_fn(row)
                if self.response_formatting_fn
                else str(row[self.answer_col]),
                self._internal_embedding_col: self.embedding_formatting_fn(row)
                if self.embedding_formatting_fn
                else query_text,
            }

        return dataset.map(_apply_format, desc="Applying formatters")

    def find_neighbors(self, dataset: Dataset, keep_rows: Optional[List[int]] = None):
        """Step 2: Generates embeddings and finds nearest neighbors.

        Embeddings are computed for every row (in dataset order), but only
        `keep_rows` (default: all rows) are indexed and queried, so rows outside
        `keep_rows` (exact duplicates) never appear as neighbors. Returned labels
        are dataset row indices with shape (len(dataset), k); rows outside
        `keep_rows` are filled with -1.
        """
        n_rows = len(dataset)
        keep = np.arange(n_rows) if keep_rows is None else np.asarray(keep_rows, dtype=np.int64)
        texts = dataset[self._internal_embedding_col]
        texts_sha256 = hashlib.sha256(
            json.dumps([str(t) for t in texts]).encode("utf-8")
        ).hexdigest()

        if self.output_neighbors_path and os.path.exists(self.output_neighbors_path):
            data = np.load(self.output_neighbors_path)
            if (
                "keep_rows" in data
                and "embedding_model" in data
                and "k_neighbors" in data
                and "texts_sha256" in data
                and str(data["texts_sha256"]) == texts_sha256
                and data["labels"].shape[0] == n_rows
                and np.array_equal(data["keep_rows"], keep)
                and str(data["embedding_model"]) == self.embedding_model_name
                and int(data["k_neighbors"]) == self.k_neighbors
            ):
                print(
                    f"Loading pre-computed neighbors from {self.output_neighbors_path}..."
                )
                return data["labels"], data["distances"]
            print(
                f"Pre-computed neighbors at {self.output_neighbors_path} do not match"
                " the current dataset; recomputing."
            )

        embeddings = None
        meta_path = (
            self.output_embeddings_path + ".meta.json"
            if self.output_embeddings_path
            else None
        )
        expected_meta = {
            "embedding_model": self.embedding_model_name,
            "n_rows": n_rows,
            "texts_sha256": texts_sha256,
        }
        if self.output_embeddings_path and os.path.exists(self.output_embeddings_path):
            meta = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            if meta == expected_meta:
                print(
                    f"Loading pre-computed embeddings from {self.output_embeddings_path}..."
                )
                embeddings = np.load(self.output_embeddings_path)
            else:
                print(
                    f"Pre-computed embeddings at {self.output_embeddings_path} were made"
                    f" for {meta}, expected {expected_meta}; recomputing."
                )
        if embeddings is None:
            print("Generating embeddings...")
            embeddings = get_embeddings(
                texts,
                model_name=self.embedding_model_name,
                batch_size=self.embedding_batch_size,
            )

            if self.output_embeddings_path:
                print(f"Saving embeddings to {self.output_embeddings_path}...")
                np.save(self.output_embeddings_path, embeddings)
                with open(meta_path, "w") as f:
                    json.dump(expected_meta, f)

        print("Building HNSW index and finding neighbors...")
        kept_embeddings = embeddings[keep]
        k = min(self.k_neighbors, len(keep))
        index = build_hnsw_index(kept_embeddings)
        sub_labels, sub_distances = find_neighbors(index, kept_embeddings, k=k)

        labels = np.full((n_rows, k), -1, dtype=np.int64)
        distances = np.full((n_rows, k), np.nan, dtype=np.float32)
        labels[keep] = keep[sub_labels.astype(np.int64)]
        distances[keep] = sub_distances

        if self.output_neighbors_path:
            print(f"Saving neighbors to {self.output_neighbors_path}...")
            np.savez(
                self.output_neighbors_path,
                labels=labels,
                distances=distances,
                keep_rows=keep,
                embedding_model=np.array(self.embedding_model_name),
                k_neighbors=np.array(self.k_neighbors),
                texts_sha256=np.array(texts_sha256),
            )

        return labels, distances

    def generate_inputs(
        self, dataset: Dataset, labels, keep_rows: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Step 3: Generates prompts for scoring.

        Inputs are always regenerated; `output_scoring_inputs_path`, if set, is
        overwritten with them.
        """
        print("Generating vLLM inputs for scoring...")
        records = generate_dataset_inputs(
            dataset, labels, self._internal_text_col, self._internal_answer_col,
            score_difference=self.score_difference,
            query_rows=keep_rows,
        )

        if self.output_scoring_inputs_path:
            print(f"Saving scoring inputs to {self.output_scoring_inputs_path}...")
            with open(self.output_scoring_inputs_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        return records

    def score_inputs(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 4: Scores the inputs using vLLM.

        The vLLM engine is only started if some records still need scoring, so
        rerunning a finished run (e.g. to redo arbitration) does not load the model.
        """
        engine = {}

        def _start_engine():
            print("Initializing vLLM...")
            engine["llm"] = LLM(**self.vllm_config)
            return engine["llm"], engine["llm"].get_tokenizer()

        scored_records = score_inputs(
            None,
            None,
            records,
            output_path=self.output_scored_records_path,
            prompt_logprobs=self.prompt_logprobs,
            llm_factory=_start_engine,
        )

        if self.score_difference:
            print("Combining scores for difference mode...")
            combined_records = {}
            for r in scored_records:
                key = (r["query_index"], r["is_base"], r["neighbor_index"])
                if key not in combined_records:
                    combined_records[key] = {}
                if r["is_chosen"]:
                    combined_records[key]["chosen_logprob"] = r["logprob"]
                    combined_records[key]["chosen_record"] = r
                else:
                    combined_records[key]["rejected_logprob"] = r["logprob"]

            final_scored_records = []
            for key, data in combined_records.items():
                if "chosen_logprob" not in data or "rejected_logprob" not in data:
                    raise ValueError(
                        "score_difference: missing the "
                        f"{'chosen' if 'chosen_logprob' not in data else 'rejected'}"
                        f" score for (query_index, is_base, neighbor_index)={key}."
                    )
                diff_lp = data["chosen_logprob"] - data["rejected_logprob"]

                res_record = data["chosen_record"]
                res_record["logprob"] = diff_lp
                res_record["probability_pct"] = np.exp(diff_lp) * 100
                final_scored_records.append(res_record)

            scored_records = final_scored_records

        if engine:
            # The vLLM model is no longer needed; free GPU memory.
            engine.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return scored_records

    def build_graph(self, scored_records: List[Dict[str, Any]]) -> QueryInfluenceGraph:
        """Step 5: Builds the QueryInfluenceGraph from the scores."""
        print(
            f"Building QueryInfluenceGraph from {len(scored_records)} records in memory..."
        )
        query_graph, errors = build_query_graph_from_records(
            records=scored_records,
            errors_output_path=self.errors_output_path,
            length_normalization=self.length_normalization,
        )

        if errors:
            print(
                f"Encountered {len(errors)} errors during graph construction. See {self.errors_output_path}"
            )
        return query_graph

    def save_graph(self, query_graph: QueryInfluenceGraph):
        """Step 6: Saves the graph to disk."""
        print(f"Saving final QueryInfluenceGraph to {self.output_graph_path}...")
        query_graph.save_to_jsonl(self.output_graph_path)

    def run(
        self,
        dataset: Optional[Dataset] = None,
        dataset_load_fn: Optional[Callable[[], Dataset]] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        skip_scoring: bool = False,
    ) -> QueryInfluenceGraph:
        """
        Executes the full pipeline.
        You can either pass a pre-loaded `dataset`, or provide `dataset_kwargs`
        which will be unpacked into `self.load_data()`, or provide a `dataset_load_fn`
        callable.

        Args:
            dataset: An optional, pre-loaded huggingface Dataset object.
            dataset_load_fn: An optional function that returns a Dataset object.
            dataset_kwargs: A dictionary of arguments to pass to `self.load_data()`.
                Required if `dataset` is not provided.
            sample_size: If provided, the pipeline will only run on a sample of
                this size from the dataset. Ideal for test runs.
            skip_scoring: If True, skips the vLLM scoring step and assigns placeholder
                scores to construct the graph structure quickly.
        """
        pipeline_start_time = time.time()

        step_start = time.time()
        if dataset is not None:
            print("Using provided dataset.")
        elif dataset_load_fn is not None:
            print("Loading dataset via provided function...")
            dataset = dataset_load_fn()
        elif dataset_kwargs is not None:
            dataset = self.load_data(**dataset_kwargs)
        else:
            raise ValueError(
                "Must provide 'dataset', 'dataset_load_fn', or 'dataset_kwargs'."
            )
        print(
            f"Step 1 (Load Dataset) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        if sample_size is not None:
            num_records = min(sample_size, len(dataset))
            print(f"Using a sample of {num_records} records for this run.")
            dataset = dataset.select(range(num_records))

        step_start = time.time()
        dataset = self.format_dataset(dataset)
        print(
            f"Step 1.5 (Format Dataset) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        # Graph nodes are keyed by (query, response) text: score only the first
        # occurrence of each exact duplicate so it cannot exemplify itself.
        keep_rows = first_occurrence_rows(
            dataset[self._internal_text_col], dataset[self._internal_answer_col]
        )
        n_duplicates = len(dataset) - len(keep_rows)
        if n_duplicates:
            print(
                f"Skipping {n_duplicates} exact duplicate (query, response) rows;"
                f" {len(keep_rows)} unique rows remain.\n"
            )

        step_start = time.time()
        labels, _ = self.find_neighbors(dataset, keep_rows)
        print(
            f"Step 2 (Embeddings & Neighbors) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        step_start = time.time()
        records = self.generate_inputs(dataset, labels, keep_rows)
        print(
            f"Step 3 (Generate Inputs) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        step_start = time.time()
        if skip_scoring:
            print(
                "Skipping vLLM scoring. Assigning placeholder scores of 0.0 to all records..."
            )
            print(
                "Exemplified/exemplifying metrics of this graph are not meaningful without scoring."
            )
            scored_records = records
            for r in scored_records:
                r["logprob"] = 0.0
                r["probability_pct"] = 100.0
        else:
            scored_records = self.score_inputs(records)
        print(
            f"Step 4 (vLLM Scoring) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        step_start = time.time()
        query_graph = self.build_graph(scored_records)
        print(
            f"Step 5 (Build graph) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        step_start = time.time()
        self.save_graph(query_graph)
        print(
            f"Step 6 (Save graph) completed in {time.time() - step_start:.2f} seconds.\n"
        )

        print(
            f"Data valuation pipeline complete. Total execution time: {time.time() - pipeline_start_time:.2f} seconds."
        )
        return query_graph
