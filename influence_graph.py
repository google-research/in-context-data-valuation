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

"""Query influence graph for data valuation.

This class is used to represent the neighboring relationships between queries
in a graph. Each node in the graph represents a query, and the edge represents
the exemplification relationship. We calculate various scores and ratios to
measure the value of each query.
"""

from concurrent import futures
import dataclasses
import itertools
import json
import threading
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import tqdm.auto as tqdm_lib

import time

@dataclasses.dataclass(slots=True)
class _QueryLogLikelihoodScore:
    """Log likelihood score of a query and a neighbor query."""

    query: str
    response: str
    score: float

@dataclasses.dataclass(slots=True)
class _QueryNode:
    """A node in the QueryInfluenceGraph.

    Each node corresponds to a unique query and stores various scores and
    relationships.

    Exemplified queries of a query stored in the node (Forward Neighbors / Teachers):
      prompt: <neighbor query> <neighbor response> <current query>
      response: <current query response>

    Exemplifying queries of a query stored in the node (Reverse Neighbors / Students):
      prompt: <current query> <current query response> <neighbor query>
      response: <neighbor query response>

    Attributes:
      query: The current query string.
      query_index: Unique integer index of the query.
      response: The response associated with the query.
      base_score: The log likelihood of the response given the query.
      exemplified_scores: A list of `_QueryLogLikelihoodScore` for queries that
        act as examples for this query (i.e., the forward k-NN neighborhood, or 
        "teachers"). The score represents the likelihood of this query's response 
        given the neighbor as an example.
      exemplifying_scores: A list of `_QueryLogLikelihoodScore` for queries that
        retrieve this query as an example (i.e., the reverse k-NN neighborhood, or 
        "students"). The score represents the likelihood of the neighbor query's 
        response given this query as an example.
      exemplified_score_ratio: The average difference between the scores in
        `exemplified_scores` and the `base_score`. This indicates how much better
        this query performs when exemplified by others. None if `base_score` is
        None or `exemplified_scores` is empty.
      exemplifying_score_ratio: The average difference between the scores in
        `exemplifying_scores` and the base scores of the neighbor queries. This
        indicates how much better other queries perform when exemplified by this
        query. None if `exemplifying_scores` is empty or base scores are missing.
      one_of_a_kind_ratio: Calculated as 1.0 - (len(exemplifying_scores) /
        len(exemplified_scores)). This ratio defines how unique the query is.
        None if `exemplified_scores` is empty. It's 1.0 if
        `exemplifying_scores` is empty.
      exemplified_advantage: Z-score measuring how much better the
        `exemplified_score_ratio` of this query is compared to the
        `exemplified_score_ratio` of the queries that exemplify it. None if
        calculation is not possible. (Population: Forward Neighbors / Teachers)
      exemplified_mean: The mean used in the `exemplified_advantage` calculation.
      exemplified_std: The standard deviation used in the `exemplified_advantage`
        calculation.
      exemplifying_advantage: Z-score measuring how much better the
        `exemplifying_score_ratio` of this query is compared to the
        `exemplifying_score_ratio` of the queries that exemplify it. None if
        calculation is not possible. (Population: Forward Neighbors / Teachers)
      exemplifying_mean: The mean used in the `exemplifying_advantage`
        calculation.
      exemplifying_std: The standard deviation used in the
        `exemplifying_advantage` calculation.
      response_num_tokens: Number of tokens in `response` as scored by the
        model. Used to normalise score differences per token when the graph's
        `length_normalization` is "tokens". None if not recorded.
    """

    query: str
    query_index: int = 0
    response: str = ""
    base_score: float | None = None
    exemplified_scores: list[_QueryLogLikelihoodScore] = dataclasses.field(
        default_factory=list
    )
    exemplifying_scores: list[_QueryLogLikelihoodScore] = dataclasses.field(
        default_factory=list
    )
    exemplified_score_ratio: float | None = None
    exemplifying_score_ratio: float | None = None
    one_of_a_kind_ratio: float | None = None
    exemplified_advantage: float | None = None
    exemplified_mean: float | None = None
    exemplified_std: float | None = None
    exemplifying_advantage: float | None = None
    exemplifying_mean: float | None = None
    exemplifying_std: float | None = None
    response_num_tokens: int | None = None

LENGTH_NORMALIZATIONS = ("none", "tokens", "chars")

def _serialize_chunk(nodes: List[_QueryNode]) -> List[str]:
    """Serializes a chunk of QueryNodes to JSONL strings."""
    lines = []
    for node in nodes:
        node_dict = dataclasses.asdict(node)

        node_dict["exemplified_scores"] = [
            dataclasses.asdict(s) for s in node.exemplified_scores
        ]
        node_dict["exemplifying_scores"] = [
            dataclasses.asdict(s) for s in node.exemplifying_scores
        ]
        lines.append(json.dumps(node_dict))
    return lines

def _save_shard_task(args: Tuple[str, List[_QueryNode], int]) -> None:
    """Serializes and writes a list of QueryNodes to a JSONL file."""
    file_name, shard_nodes, chunk_size = args
    with open(file_name, "w") as f_out:
        for i in range(0, len(shard_nodes), chunk_size):
            chunk = shard_nodes[i : i + chunk_size]
            for line in _serialize_chunk(chunk):
                f_out.write(line + "\n")

class QueryInfluenceGraph:
    """Directed influence graph for query relationships.

    This class stores and manages relationships between queries based on
    exemplification, including scores and calculated ratios/advantages.
    """

    def __init__(
        self,
        enable_partial_update: bool = True,
        length_normalization: str = "none",
    ):
        """Initialises an empty graph.

        Args:
          enable_partial_update: Recompute ratios lazily for touched nodes.
          length_normalization: How each edge's score difference
            (score - student's base score) is scaled before averaging into the
            ratios. "none" leaves it in total nats over the whole response;
            "tokens" divides by the student's response token count (per-token
            nats); "chars" divides by its character count, for graphs whose
            records carry no token counts.
        """
        if length_normalization not in LENGTH_NORMALIZATIONS:
            raise ValueError(
                f"length_normalization must be one of {LENGTH_NORMALIZATIONS}, "
                f"got {length_normalization!r}"
            )
        self.nodes: Dict[str, Dict[str, _QueryNode]] = {}
        self._dirty_queries: Set[str] = set()
        self.enable_partial_update = enable_partial_update
        self.length_normalization = length_normalization
        self._warned_missing_tokens = False

    def _length_divisor(self, node: _QueryNode) -> float:
        """Returns the divisor applied to score differences for `node`'s response."""
        if self.length_normalization == "none":
            return 1.0
        if self.length_normalization == "tokens":
            if node.response_num_tokens:
                return float(node.response_num_tokens)
            if not self._warned_missing_tokens:
                print(
                    "[QueryInfluenceGraph] length_normalization='tokens' but some "
                    "nodes carry no response_num_tokens; falling back to character "
                    "count for those nodes.",
                    flush=True,
                )
                self._warned_missing_tokens = True
        return float(max(1, len(node.response)))

    @staticmethod
    def _load_jsonl_file(file_path: str) -> List[_QueryNode]:
        """Deserializes a single JSONL file into a list of QueryNodes."""
        nodes = []
        with open(file_path, "r") as f_in_file:
            t_start = time.time()
            lines = f_in_file.read().splitlines()
            t_read = time.time() - t_start

            t_json = 0.0
            t_classes = 0.0

            for line in lines:
                if not line.strip():
                    continue

                t_json_start = time.time()
                data = json.loads(line)
                t_json += time.time() - t_json_start

                t_class_start = time.time()
                data["exemplified_scores"] = [
                    _QueryLogLikelihoodScore(**d)
                    for d in data.get("exemplified_scores", [])
                ]
                data["exemplifying_scores"] = [
                    _QueryLogLikelihoodScore(**d)
                    for d in data.get("exemplifying_scores", [])
                ]

                nodes.append(_QueryNode(**data))
                t_classes += time.time() - t_class_start

        print(
            f"  [Worker] file={file_path.split('/')[-1]} read={t_read:.2f}s json={t_json:.2f}s dataclass={t_classes:.2f}s",
            flush=True,
        )
        return nodes

    def _get_node(
        self, query: str, response: str, strict: bool = False
    ) -> _QueryNode | None:
        query_dict = self.nodes.get(query)
        if query_dict and response in query_dict:
            return query_dict[response]
        if strict:
            raise KeyError(
                f"Query '{query}' with response '{response}' not found in graph."
            )
        return None

    def _get_or_create_node(self, query: str, response: str) -> _QueryNode:
        query_dict = self.nodes.get(query)
        if query_dict is None:
            query_dict = {}
            self.nodes[query] = query_dict

        if response not in query_dict:
            query_dict[response] = _QueryNode(query=query, response=response)
        return query_dict[response]

    def recalculate_all_nodes(self) -> None:
        """Forces recalculation of posterior ratios and advantages for the entire graph."""
        all_queries = set(self.nodes.keys())
        self._calc_posterior_ratios(all_queries)
        self._calc_posterior_advantages(all_queries)
        self._dirty_queries.clear()

    def _recalculate_dirty_nodes(self) -> None:
        """Calculates posterior ratios and advantages for dirty nodes."""
        if not self._dirty_queries:
            return

        # 1. Ratios only depend on the base scores of a node and its exemplified
        # or exemplifying neighbors. Therefore, we only need to recalculate ratios
        # for the dirty queries.
        self._calc_posterior_ratios(self._dirty_queries)

        # 2. Advantages depend on the RATIOS of a node's neighbors.
        # Since the ratios for the dirty queries just changed, we must update
        # the advantage not just for the dirty queries, but for any query
        # connected to them.
        advantage_dirty_queries = set(self._dirty_queries)
        for q in self._dirty_queries:
            query_dict = self.nodes.get(q)
            if query_dict:
                for node in query_dict.values():
                    for score in node.exemplified_scores:
                        advantage_dirty_queries.add(score.query)
                    for score in node.exemplifying_scores:
                        advantage_dirty_queries.add(score.query)

        self._calc_posterior_advantages(advantage_dirty_queries)

        self._dirty_queries.clear()

    def check_clean_for_save(self) -> None:
        """Verifies that all math is up to date before saving the graph."""
        if self._dirty_queries:
            raise RuntimeError(
                f"Cannot save graph: {len(self._dirty_queries)} dirty queries exist. "
                "Mathematical properties (like ratios and advantages) are out of date. "
                "Call a getter to force recalculation before saving, or trigger "
                "_recalculate_dirty_nodes()."
            )

    def save_to_jsonl(
        self,
        file_path: str,
        num_workers: int = 16,
        chunk_size: int = 10000,
    ):
        """Saves the graph to a JSONL file in parallel.

        Args:
          file_path: The local path where the JSONL file will be written.
          num_workers: The number of parallel workers to use for serialization.
          chunk_size: The number of nodes to serialize in a single batch.

        """
        try:
            self.check_clean_for_save()
        except RuntimeError as e:
            print(e)
            print("Recalculating dirty nodes...")
            self._recalculate_dirty_nodes()

        file_names = [file_path]

        nodes = []
        for query_dict in self.nodes.values():
            nodes.extend(query_dict.values())

        if len(file_names) > 1:
            shard_size = (len(nodes) + len(file_names) - 1) // len(file_names)
            if shard_size == 0:
                shard_size = 1

            tasks = []
            for i, file_name in enumerate(file_names):
                shard_nodes = nodes[i * shard_size : (i + 1) * shard_size]
                tasks.append((file_name, shard_nodes, chunk_size))

            with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for _ in tqdm_lib.tqdm(
                    executor.map(_save_shard_task, tasks),
                    total=len(tasks),
                    desc=f"Saving {len(nodes)} nodes to {len(file_names)} shards",
                ):
                    pass
        else:
            with open(file_names[0], "w") as f_out:
                nodes_iter = iter(nodes)
                with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    while True:
                        chunks = []
                        for _ in range(num_workers * 2):
                            chunk = list(itertools.islice(nodes_iter, chunk_size))
                            if not chunk:
                                break
                            chunks.append(chunk)

                        if not chunks:
                            break

                        for serialized_chunk in tqdm_lib.tqdm(
                            executor.map(_serialize_chunk, chunks),
                            total=len(chunks),
                            desc="Serializing chunks",
                        ):
                            for line in serialized_chunk:
                                f_out.write(line + "\n")

    @classmethod
    def load_from_jsonl(
        cls,
        file_path: str,
        num_workers: int = 16,
        max_pending_tasks: int = 300,
        length_normalization: str | None = None,
    ) -> "QueryInfluenceGraph":
        """Loads a graph from a JSONL file in parallel.

        Args:
          file_path: The local path where the JSONL file is located.
          num_workers: The number of parallel workers to use for deserialization.
          max_pending_tasks: The maximum number of pending futures allowed before
            blocking the thread pool submission.
          length_normalization: If given, all ratios and advantages stored in the
            file are discarded and recomputed under this normalisation (see
            `__init__`). If None, the stored values are used as-is.

        Returns:
          A fully constructed QueryInfluenceGraph.
        """
        graph = cls(length_normalization=length_normalization or "none")

        file_names = [file_path]

        semaphore = threading.Semaphore(max_pending_tasks)

        def task_complete_callback(future):
            """Callback function to release the semaphore when a task is done."""
            del future
            semaphore.release()

        def submit_task_with_limit(executor, fn, *args, **kwargs):
            """Acquires a semaphore permit before submitting the task."""
            semaphore.acquire()  # Blocks if the limit is reached
            future = executor.submit(fn, *args, **kwargs)
            future.add_done_callback(task_complete_callback)
            return future

        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            tasks = []
            for f in file_names:
                future = submit_task_with_limit(executor, cls._load_jsonl_file, f)
                tasks.append(future)

            time_future_wait = 0.0
            time_dict_update = 0.0

            for future in tqdm_lib.tqdm(
                futures.as_completed(tasks),
                total=len(tasks),
                desc="Loading nodes from JSONL shards",
            ):
                start_wait = time.time()
                result = future.result()
                time_future_wait += time.time() - start_wait

                start_update = time.time()
                for node in result:
                    query_dict = graph.nodes.get(node.query)
                    if query_dict is None:
                        query_dict = {}
                        graph.nodes[node.query] = query_dict
                    query_dict[node.response] = node
                time_dict_update += time.time() - start_update

        print("\n[Metrics] load_from_jsonl timings:", flush=True)
        print(f"  - IPC / Worker Wait Time: {time_future_wait:.2f} seconds", flush=True)
        print(
            f"  - Main Thread Dictionary Integration Time: {time_dict_update:.2f} seconds\n",
            flush=True,
        )

        if length_normalization is not None:
            print(
                f"Recomputing ratios/advantages with length_normalization="
                f"{length_normalization!r}...",
                flush=True,
            )
            graph.recalculate_all_nodes()

        return graph

    def add_base_query(
        self,
        query: str,
        query_index: int,
        response: str,
        score: float,
        response_num_tokens: int | None = None,
    ) -> None:
        """Adds a base query to the graph.

        Args:
          query: The current query string.
          query_index: The integer index of the current query.
          response: The response associated with the current query.
          score: The likelihood score of the response for the current query.
          response_num_tokens: Number of tokens in `response` as scored, if known.

        Raises:
          ValueError: If the query already exists with a different index.
        """
        node = self._get_or_create_node(query, response)

        if node.base_score is not None:
            if node.query_index != query_index:
                raise ValueError(
                    f"Query '{query}' with response '{response}' already exists with a"
                    f" different index. Existing: index={node.query_index}. New:"
                    f" index={query_index}."
                )

        node.query_index = query_index
        node.base_score = score
        if response_num_tokens is not None:
            node.response_num_tokens = int(response_num_tokens)
        if self.enable_partial_update:
            self._dirty_queries.add(query)
            for score in node.exemplified_scores:
                self._dirty_queries.add(score.query)

    def add_exemplification(
        self,
        query: str,
        response: str,
        example_query: str,
        example_response: str,
        score: float,
    ) -> None:
        """Adds an exemplification relationship to the graph.

        Signifies that `query` is being exemplified by `example_query`.
        The `score` represents the likelihood of the `query` given the
        `example_query`.
        The relationships are stored in `exemplified_scores` (what exemplifies `query`, 
        i.e. its teachers) and `exemplifying_scores` (what `query` exemplifies, i.e. 
        its students).

        Args:
          query: The current query string.
          response: The response associated with the current query.
          example_query: The query string used as an example.
          example_response: The response associated with the example query.
          score: The likelihood score of the query given the example_query.
        """
        node = self._get_or_create_node(query, response)
        ex_node = self._get_or_create_node(example_query, example_response)

        node.exemplified_scores.append(
            _QueryLogLikelihoodScore(
                query=example_query, response=example_response, score=score
            )
        )
        ex_node.exemplifying_scores.append(
            _QueryLogLikelihoodScore(query=query, response=response, score=score)
        )
        if self.enable_partial_update:
            self._dirty_queries.add(query)
            self._dirty_queries.add(example_query)

    def merge_graph(
        self, graph: "QueryInfluenceGraph", strict: bool = True
    ) -> List[Dict[str, Any]]:
        """Merges another QueryInfluenceGraph into the current graph.

        Args:
          graph: The graph to merge into this one.
          strict: If True, raises ValueError on collision. If False, captures the error
            and continues merging remaining non-conflicting nodes.

        Returns:
          A list of dictionaries characterizing node collisions if strict=False.
        """
        failed_merges = []

        for query, query_dict in graph.nodes.items():
            for other_response, other_node in query_dict.items():
                node = self._get_or_create_node(query, other_response)
                try:
                    if other_node.base_score is not None:
                        if node.base_score is not None:
                            if node.query_index != other_node.query_index:
                                raise ValueError(
                                    f"Query '{query}' with response '{other_node.response}' from merged graph conflicts with existing index."
                                    f" Existing: index={node.query_index}."
                                    f" Merged: index={other_node.query_index}."
                                )

                        node.base_score = other_node.base_score
                        node.query_index = other_node.query_index
                        if other_node.response_num_tokens is not None:
                            node.response_num_tokens = other_node.response_num_tokens
                    elif other_node.response:
                        if not node.response:
                            node.query_index = other_node.query_index

                    if other_node.exemplified_scores:
                        node.exemplified_scores.extend(other_node.exemplified_scores)
                    if other_node.exemplifying_scores:
                        node.exemplifying_scores.extend(other_node.exemplifying_scores)
                except ValueError as e:
                    if strict:
                        raise e
                    failed_merges.append(
                        {
                            "query": query,
                            "response": other_node.response,
                            "error": str(e),
                        }
                    )

        if self.enable_partial_update:
            for query in graph.nodes:
                self._dirty_queries.add(query)

        return failed_merges

    def _calc_posterior_ratios(self, queries: Set[str]):
        """Calculates and stores various posterior ratios for each query.

        This method computes three types of ratios for each query in the graph:
        - `exemplified_score_ratio`: The average difference between the scores
          of this query when exemplified by others, and this query's base score.
        - `exemplifying_score_ratio`: The average difference between the scores
          of queries that this query exemplifies, and their respective base scores.
        - `one_of_a_kind_ratio`: A ratio indicating how unique a query is, based on
          the number of queries it exemplifies versus the number of queries that
          exemplify it.

        Each score difference is divided by the length of the response whose
        likelihood was measured, as set by `length_normalization`.

        Args:
          queries: A set of queries for which to calculate the ratios.
        """
        for query in tqdm_lib.tqdm(queries, desc="Calculating posterior ratios"):
            query_dict = self.nodes.get(query)
            if not query_dict:
                continue
            for node in query_dict.values():
                if node.base_score is None or len(node.exemplified_scores) == 0:
                    node.exemplified_score_ratio = None
                else:
                    own_div = self._length_divisor(node)
                    node.exemplified_score_ratio = sum(
                        (score.score - node.base_score) / own_div
                        for score in node.exemplified_scores
                    ) / len(node.exemplified_scores)

                if len(node.exemplifying_scores) == 0:
                    node.exemplifying_score_ratio = None
                else:
                    ratios = []
                    for example in node.exemplifying_scores:
                        ex_node = self._get_node(example.query, example.response)
                        if ex_node is None or ex_node.base_score is None:
                            continue
                        ratios.append(
                            (example.score - ex_node.base_score)
                            / self._length_divisor(ex_node)
                        )
                    node.exemplifying_score_ratio = np.mean(ratios) if ratios else None

                # If the node has no incoming edges, its uniqueness relative to other
                # queries in the neighborhood is undefined (division by zero).
                if len(node.exemplified_scores) == 0:
                    node.one_of_a_kind_ratio = None
                # If the node has incoming edges but no outgoing edges,
                # its one-of-a-kind ratio is 1.
                elif len(node.exemplifying_scores) == 0:
                    node.one_of_a_kind_ratio = 1.0
                else:
                    node.one_of_a_kind_ratio = 1.0 - len(
                        node.exemplifying_scores
                    ) / len(node.exemplified_scores)

    def _calc_posterior_advantages(self, queries: Set[str]) -> None:
        """Calculates and stores advantage scores for each query."""
        for query in tqdm_lib.tqdm(queries, desc="Calculating posterior advantages"):
            query_dict = self.nodes.get(query)
            if not query_dict:
                continue
            for node in query_dict.values():
                base_ratio = node.exemplified_score_ratio
                ann_ratios = []
                for example in node.exemplified_scores:
                    ex_node = self._get_node(example.query, example.response)
                    if ex_node and ex_node.exemplified_score_ratio is not None:
                        ann_ratios.append(ex_node.exemplified_score_ratio)

                if base_ratio is None or len(ann_ratios) == 0:
                    node.exemplified_advantage = None
                    node.exemplified_mean = None
                    node.exemplified_std = None
                else:
                    node.exemplified_mean = float(np.mean(ann_ratios + [base_ratio]))
                    std_val = float(np.std(ann_ratios + [base_ratio], ddof=1))
                    node.exemplified_std = std_val
                    if std_val:
                        node.exemplified_advantage = (
                            base_ratio - node.exemplified_mean
                        ) / std_val
                    else:
                        node.exemplified_advantage = 0.0

                # calc exemplifying advantages: each incoming neighbor's reference is its
                # mean outgoing score ratio excluding the edge that targets `node`,
                # so `node`'s own target difficulty does not contaminate its neighborhood reference.
                base_ratio = node.exemplifying_score_ratio
                ann_ratios = []
                for example in node.exemplified_scores:
                    ex_node = self._get_node(example.query, example.response)
                    if not ex_node or not ex_node.exemplifying_scores:
                        continue
                    rem_ratios = []
                    for out_edge in ex_node.exemplifying_scores:
                        if out_edge.query == node.query and out_edge.response == node.response:
                            continue
                        target_node = self._get_node(out_edge.query, out_edge.response)
                        if target_node is None or target_node.base_score is None:
                            continue
                        rem_ratios.append(
                            (out_edge.score - target_node.base_score)
                            / self._length_divisor(target_node)
                        )
                    if rem_ratios:
                        ann_ratios.append(float(np.mean(rem_ratios)))

                if base_ratio is None or len(ann_ratios) == 0:
                    node.exemplifying_advantage = None
                    node.exemplifying_mean = None
                    node.exemplifying_std = None
                else:
                    node.exemplifying_mean = float(np.mean(ann_ratios + [base_ratio]))
                    std_val = float(np.std(ann_ratios + [base_ratio], ddof=1))
                    node.exemplifying_std = std_val
                    if std_val:
                        node.exemplifying_advantage = (
                            base_ratio - node.exemplifying_mean
                        ) / std_val
                    else:
                        node.exemplifying_advantage = 0.0

    def get_base_score(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        node = self._get_node(query, response, strict=strict)
        return node.base_score if node else None

    def get_exemplified_scores(
        self, query: str, response: str, *, strict: bool = False
    ) -> list[_QueryLogLikelihoodScore] | None:
        node = self._get_node(query, response, strict=strict)
        return node.exemplified_scores if node else None

    def get_exemplifying_scores(
        self, query: str, response: str, *, strict: bool = False
    ) -> list[_QueryLogLikelihoodScore] | None:
        node = self._get_node(query, response, strict=strict)
        return node.exemplifying_scores if node else None

    def get_exemplified_score_ratio(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        self._recalculate_dirty_nodes()
        node = self._get_node(query, response, strict=strict)
        return node.exemplified_score_ratio if node else None

    def get_exemplifying_score_ratio(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        self._recalculate_dirty_nodes()
        node = self._get_node(query, response, strict=strict)
        return node.exemplifying_score_ratio if node else None

    def get_one_of_a_kind_ratio(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        self._recalculate_dirty_nodes()
        node = self._get_node(query, response, strict=strict)
        return node.one_of_a_kind_ratio if node else None

    def get_exemplified_advantage(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        self._recalculate_dirty_nodes()
        node = self._get_node(query, response, strict=strict)
        return node.exemplified_advantage if node else None

    def get_exemplifying_advantage(
        self, query: str, response: str, *, strict: bool = False
    ) -> float | None:
        self._recalculate_dirty_nodes()
        node = self._get_node(query, response, strict=strict)
        return node.exemplifying_advantage if node else None

def _parse_record_structure(parsed_record: Dict[str, Any]) -> Tuple[str, int, str | None, str | None]:
    """Extracts (query, query_index, example_query, example_response) from a record."""
    query = str(parsed_record["query_text"]).strip()
    query_index = int(parsed_record["query_index"])
    example_query = parsed_record.get("example_query")
    example_response = parsed_record.get("example_response")
    if example_query is not None:
        example_query = str(example_query).strip()
    if example_response is not None:
        example_response = str(example_response).strip()
    return query, query_index, example_query, example_response

def _build_graph_from_record_stream(
    record_iterator, source_name: str = "stream"
) -> Tuple[QueryInfluenceGraph, List[Dict[str, Any]]]:
    """Builds a QueryInfluenceGraph from an iterator of scored records.

    Args:
      record_iterator: An iterator that yields scored record dictionaries.
      source_name: A string identifier for the source, used for logging.

    Returns:
      A tuple containing the populated QueryInfluenceGraph and a list of records
      that failed to process.
    """
    query_graph = QueryInfluenceGraph(enable_partial_update=False)
    failed_records = []
    count = 0

    for parsed_record in record_iterator:
        count += 1

        try:
            response = parsed_record["scoring_completions"].strip()
            if "scores" in parsed_record:
                score = float(parsed_record["scores"])
            elif "logprob" in parsed_record:
                score = float(parsed_record["logprob"])
            else:
                raise KeyError("Record must contain either 'scores' or 'logprob' key.")

            query, query_index, example_query, example_response = (
                _parse_record_structure(parsed_record)
            )

            if example_query is None:
                query_graph.add_base_query(
                    query=query,
                    query_index=query_index,
                    response=response,
                    score=score,
                    response_num_tokens=parsed_record.get("completion_num_tokens"),
                )
            elif parsed_record.get("prompt_truncated"):
                # A truncated 1-shot prompt lost its instruction and the start of
                # its exemplar, so its score is not a valid edge weight.
                raise ValueError(
                    "1-shot prompt was truncated to fit the model context; edge skipped."
                )
            else:
                # To handle cases where a node is seen as an example before its base score is processed.
                node = query_graph._get_or_create_node(query, response)
                if node.base_score is not None:
                    if node.query_index != query_index:
                        raise ValueError(
                            "Query"
                            f" '{query}' with response '{response}' from parsed file"
                            " conflicts with existing index. Existing:"
                            f" index={node.query_index}. Parsed: index={query_index}."
                        )

                node.query_index = query_index

                query_graph.add_exemplification(
                    query=query,
                    response=response,
                    example_query=example_query,
                    example_response=example_response,
                    score=score,
                )
        except (ValueError, KeyError, IndexError) as e:
            failed_records.append({"record": parsed_record, "error": str(e)})

        if count % 10000 == 0:
            print(f"Processed {count} records from {source_name}", flush=True)

    return query_graph, failed_records

def build_query_graph_from_records(
    records: List[Dict[str, Any]],
    max_workers: int = 100,
    max_pending_tasks: int = 300,
    errors_output_path: str | None = None,
    length_normalization: str = "none",
) -> Tuple[QueryInfluenceGraph, List[Dict[str, Any]]]:
    """Builds a QueryInfluenceGraph from a list of scored records in memory.

    Args:
      records: A list of scored record dictionaries.
      max_workers: The maximum number of parallel workers.
      max_pending_tasks: The maximum number of pending futures allowed.
      errors_output_path: Optional path to write failed records to as JSONL.
      length_normalization: See `QueryInfluenceGraph.__init__`.

    Returns:
      A Tuple of (QueryInfluenceGraph, list of all failed records).
    """
    query_graph = QueryInfluenceGraph(
        enable_partial_update=False, length_normalization=length_normalization
    )
    all_errors = []

    semaphore = threading.Semaphore(max_pending_tasks)

    def task_complete_callback(future):
        del future
        semaphore.release()

    def submit_task_with_limit(executor, fn, *args, **kwargs):
        semaphore.acquire()
        future = executor.submit(fn, *args, **kwargs)
        future.add_done_callback(task_complete_callback)
        return future

    if not records:
        return query_graph, all_errors

    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        num_chunks = min(max_workers, len(records))
        chunk_size = (
            (len(records) + num_chunks - 1) // num_chunks if num_chunks > 0 else 0
        )

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            future = submit_task_with_limit(
                executor, _build_graph_from_record_stream, iter(chunk), f"chunk_{i}"
            )
            tasks.append(future)

        for future in tqdm_lib.tqdm(
            futures.as_completed(tasks),
            total=len(tasks),
            desc="Processing record chunks",
        ):
            shard_graph, shard_errors = future.result()
            all_errors.extend(shard_errors)
            merge_errors = query_graph.merge_graph(shard_graph, strict=False)
            all_errors.extend(merge_errors)

    print("Recalculating all ratios and advantages for final merged graph...")
    query_graph.recalculate_all_nodes()

    if errors_output_path and all_errors:
        with open(errors_output_path, "w") as f_out:
            for err in all_errors:
                f_out.write(json.dumps(err) + "\n")

    return query_graph, all_errors
