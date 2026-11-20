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

import collections
import concurrent.futures
import hashlib
import json
import math
from typing import Dict, List, Any, Tuple
import litellm
from influence_graph import QueryInfluenceGraph
from tqdm.auto import tqdm

import os
import time
import sys

_BLOCKED_RETRIES = 2
_TIMEOUT_RETRIES = 3
_PARSE_RETRIES = 1

# Transient provider-side failures worth retrying with backoff.
_BACKPRESSURE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "529",
    "quota",
    "rate limit",
    "overloaded",
    "internal server error",
    "service unavailable",
    "bad gateway",
)

def _eval_cache_key(model_name: str, prompt_data: Dict[str, Any]) -> Tuple[str, str]:
    """Cache key for one arbitration call: the model plus a hash of the exact prompt."""
    prompt_text = json.dumps(
        [prompt_data.get("system_instruction"), prompt_data.get("user_prompt")]
    )
    return (model_name, hashlib.sha256(prompt_text.encode("utf-8")).hexdigest())

_INSTRUCTION = [
    "You are an expert Data Quality Evaluator. You are reviewing training records for a fine-tuning dataset to ensure consistency and correctness.",
    "",
    "You will be provided with a set of [Guidelines] (if any) and two training records ([Record 1] and [Record 2]). Each record consists of an [Input] and a [Target Response].",
    "",
    "These two records were flagged by a system because they degrade each other's score during training. This usually happens when they represent related scenarios but have contradictory or inconsistent target responses.",
    "",
    "Your task is to evaluate both records independently to find the bad data.",
    "",
    "**Step-by-Step Instructions:**",
    "1. **Determine True Intent:** Look at the raw [Input] and determine what the correct response should be based on common sense and the guidelines.",
    "2. **Identify the Shared Theme:** Are Record 1 and Record 2 testing a similar concept or edge case?",
    "3. **Independent Evaluation:** Judge Record 1 strictly against your expected correct response. Do the exact same for Record 2.",
    "4. **Resolve the Inconsistency:** Identify which record (if any) is incorrect, low quality, or inconsistent. Select an action based on which record contains the most accurate target response.",
    "",
    "* CRITICAL: Only pick a winner if one target response clearly aligns with the correct intent and the other violates it. If both are technically acceptable or both are completely wrong, put 'Neither' in the 'better_record' field.",
]

def evaluate_contradictions_with_llm(
    prompts: List[Dict[str, Any]],
    model_name: str,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_workers: int = 10,
    output_path: str | None = None,
    timeout: float | None = 60,
) -> List[Dict[str, Any]]:
    """
    Evaluates a list of contradiction prompts using the LiteLLM API.

    Args:
        prompts: Output from generate_llm_prompts_for_contradictions.
        model_name: The LiteLLM model string to use (e.g., '<provider>/<model>').
        api_key: LLM API key. If None, uses the provider's standard environment
            variables.
        temperature: Sampling temperature (dropped for models that reject it).
        max_workers: Maximum number of concurrent API requests.
        output_path: Optional file path to iteratively save results as JSONL.
        timeout: Per-request timeout in seconds passed through to litellm.

    Returns:
        A list of prompt dictionaries enriched with the 'llm_evaluation' JSON result.
        Each result also records `eval_model` and `prompt_sha256`; cached results
        in `output_path` are reused only for the same model and exact prompt.
    """

    eval_cache = {}

    if output_path and os.path.exists(output_path):
        print(f"Found existing evaluations at {output_path}. Loading progress...")
        n_error_rows = 0
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # e.g. a line cut off by an interrupted run
                    ev = record.get("llm_evaluation")
                    if isinstance(ev, list) and len(ev) == 1:
                        ev = ev[0]
                    # Skip cached rows that only contain transient API/parse errors.
                    if not isinstance(ev, dict) or (
                        "error" in ev and "verdict" not in ev and "Verdict" not in ev
                    ):
                        n_error_rows += 1
                        continue
                    eval_key = (record.get("eval_model"), record.get("prompt_sha256"))
                    eval_cache[eval_key] = record
        if n_error_rows:
            print(
                f"  {n_error_rows} previous rows were API/parse errors and will be "
                "re-evaluated."
            )

    results = []
    prompts_to_evaluate = []

    for p in prompts:
        prompt_key = _eval_cache_key(model_name, p)
        if prompt_key in eval_cache:
            record = eval_cache[prompt_key]
            results.append(record)
        else:
            prompts_to_evaluate.append(p)

    print(
        f"Evaluation progress: {len(results)} already loaded, {len(prompts_to_evaluate)} new prompts to evaluate."
    )

    def _evaluate_single(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = 10
        base_delay = 2
        max_delay = 60

        messages = []
        if prompt_data.get("system_instruction"):
            messages.append(
                {"role": "system", "content": prompt_data.get("system_instruction")}
            )
        messages.append({"role": "user", "content": prompt_data.get("user_prompt")})

        cache_model, prompt_sha256 = _eval_cache_key(model_name, prompt_data)
        parse_failures = 0
        # Optional parameters the model rejected; omitted on retry.
        omitted_params = set()

        for attempt in range(max_retries):
            try:
                completion_kwargs = dict(
                    model=model_name,
                    messages=messages,
                    timeout=timeout,
                    # Silently drop parameters a provider does not support (for
                    # example `temperature` on some reasoning models).
                    drop_params=True,
                )
                if "temperature" not in omitted_params:
                    completion_kwargs["temperature"] = temperature
                if "response_format" not in omitted_params:
                    completion_kwargs["response_format"] = {"type": "json_object"}
                if api_key:
                    completion_kwargs["api_key"] = api_key
                response = litellm.completion(**completion_kwargs)

                result = prompt_data.copy()
                result["eval_model"] = cache_model
                result["prompt_sha256"] = prompt_sha256
                response_text = response.choices[0].message.content

                if not response_text:
                    raise ValueError(
                        "Response text is None (possibly blocked by safety filters or empty)."
                    )

                clean_text = response_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]

                try:
                    parsed_json = json.loads(clean_text.strip())
                    if isinstance(parsed_json, list) and len(parsed_json) > 0:
                        parsed_json = parsed_json[0]
                    result["llm_evaluation"] = parsed_json
                except json.JSONDecodeError:
                    if parse_failures < _PARSE_RETRIES and attempt < max_retries - 1:
                        parse_failures += 1
                        continue
                    result["llm_evaluation"] = {
                        "raw_text": response_text,
                        "error": "Failed to parse JSON",
                    }

                return result
            except Exception as e:
                error_msg = str(e).lower()
                rejected = [
                    p
                    for p in ("temperature", "response_format")
                    if p in error_msg
                    and p not in omitted_params
                    and any(
                        w in error_msg
                        for w in ("deprecated", "not supported", "unsupported", "does not support")
                    )
                ]
                if rejected and attempt < max_retries - 1:
                    omitted_params.update(rejected)
                    continue
                backpressure = any(m in error_msg for m in _BACKPRESSURE_MARKERS)
                timed_out = (
                    "timeout" in error_msg
                    or "timed out" in error_msg
                    or "connection" in error_msg
                )
                blocked = "response text is none" in error_msg
                if (
                    (backpressure and attempt < max_retries - 1)
                    or (timed_out and attempt < _TIMEOUT_RETRIES)
                    or (blocked and attempt < _BLOCKED_RETRIES)
                ):
                    sleep_time = min(base_delay * (2**attempt), max_delay)
                    time.sleep(sleep_time)
                    continue

                print(f"Error calling LLM API: {e}")
                result = prompt_data.copy()
                result["eval_model"] = cache_model
                result["prompt_sha256"] = prompt_sha256
                result["llm_evaluation"] = {
                    "error": str(e),
                    "error_kind": (
                        "blocked"
                        if blocked
                        else "timeout"
                        if timed_out
                        else "backpressure"
                        if backpressure
                        else "other"
                    ),
                }
                return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(_evaluate_single, p): p for p in prompts_to_evaluate
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_prompt),
            total=len(prompts_to_evaluate),
            desc="Evaluating via LLM API",
            file=sys.stdout,
            leave=True,
        ):
            result = future.result()
            results.append(result)

            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

    return results

def find_similar_bad_neighbors(
    graph: QueryInfluenceGraph,
    bad_records: List[Dict[str, Any]],
    score_lookup: Dict[int, float] = None,
    query_embeddings: Any = None,
    prompt_similarity_threshold: float = 0,
    score_threshold: float = 2.0,
    response_similarity_threshold: float = 0.5,
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Finds neighbors of bad quality records and computes both prompt and response similarities.
    Filters the neighbors to keep only those with similarities > thresholds.

    Args:
        graph: QueryInfluenceGraph instance without scores, used for neighbor finding.
        bad_records: List of dicts representing bad quality records. Expected to have 'query' and 'response'.
        score_lookup: Optional dict mapping query_index (int) to their external scores (e.g., helpfulness).
        query_embeddings: Optional numpy array or path to .npy file containing precomputed query embeddings.
            If provided, `graph` nodes must have valid `query_index` mapping to this array.
        prompt_similarity_threshold: Minimum cosine similarity between prompts (queries) to be selected.
        score_threshold: Maximum score for a neighbor to be selected (e.g., helpfulness < 2.0).
        response_similarity_threshold: Minimum cosine similarity between responses to be selected.
        embedding_model_name: The SentenceTransformer model used for similarity.
        batch_size: Batch size for generating embeddings on the fly.

    Returns:
        A list of dicts with the bad record, its similar neighbor, and the similarity score.
    """
    from neighbor_finding import get_embeddings
    import numpy as np

    if isinstance(query_embeddings, str):
        print(f"Loading query embeddings from {query_embeddings}...")
        query_embeddings = np.load(query_embeddings)

    pairs_to_check = []
    unique_responses = set()
    unique_queries = set()
    seen_pairs = set()

    stats = {
        "total_neighbors_inspected": 0,
        "skipped_same_query": 0,
        "skipped_duplicate_pair": 0,
        "skipped_missing_node": 0,
        "skipped_score_too_high": 0,
    }

    for bad_record in bad_records:
        query = bad_record["query"]
        response = bad_record.get("response", "")

        node = graph._get_node(query, response)
        if not node:
            stats["skipped_missing_node"] += 1
            continue

        neighbors = []
        neighbors.extend(node.exemplified_scores)
        neighbors.extend(node.exemplifying_scores)

        for neighbor in neighbors:
            stats["total_neighbors_inspected"] += 1

            neighbor_query = neighbor.query
            neighbor_response = neighbor.response

            if neighbor_query == query:
                stats["skipped_same_query"] += 1
                continue

            pair_key = tuple(
                sorted([(query, response), (neighbor_query, neighbor_response)])
            )
            if pair_key in seen_pairs:
                stats["skipped_duplicate_pair"] += 1
                continue
            seen_pairs.add(pair_key)

            neighbor_node = graph._get_node(neighbor_query, neighbor_response)
            if not neighbor_node:
                stats["skipped_missing_node"] += 1
                continue

            neighbor_idx = neighbor_node.query_index
            neighbor_score = None

            if score_lookup is not None:
                neighbor_score = score_lookup.get(neighbor_idx)
                if neighbor_score is None or neighbor_score >= score_threshold:
                    stats["skipped_score_too_high"] += 1
                    continue

            unique_responses.add(response)
            unique_responses.add(neighbor_response)

            if query_embeddings is None:
                unique_queries.add(query)
                unique_queries.add(neighbor_query)

            pairs_to_check.append(
                {
                    "bad_query": query,
                    "bad_response": response,
                    "bad_query_index": node.query_index,
                    "bad_score": (
                        score_lookup.get(node.query_index)
                        if score_lookup is not None
                        else None
                    ),
                    "neighbor_query": neighbor_query,
                    "neighbor_response": neighbor_response,
                    "neighbor_query_index": neighbor_idx,
                    "neighbor_score": neighbor_score,
                }
            )

    print(f"\nNeighbor Filtering Stats:\n{json.dumps(stats, indent=2)}\n")

    if not pairs_to_check:
        return []

    unique_responses_list = list(unique_responses)
    response_to_idx = {resp: i for i, resp in enumerate(unique_responses_list)}

    print(f"Generating embeddings for {len(unique_responses_list)} unique responses...")
    resp_embeddings = get_embeddings(
        unique_responses_list, model_name=embedding_model_name, batch_size=batch_size
    )

    resp_norms = np.linalg.norm(resp_embeddings, axis=1, keepdims=True)
    resp_norms[resp_norms == 0] = 1
    resp_embeddings_normalized = resp_embeddings / resp_norms

    if query_embeddings is None:
        unique_queries_list = list(unique_queries)
        query_to_idx = {q: i for i, q in enumerate(unique_queries_list)}

        print(f"Generating embeddings for {len(unique_queries_list)} unique queries...")
        q_embeddings = get_embeddings(
            unique_queries_list, model_name=embedding_model_name, batch_size=batch_size
        )

        q_norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1
        q_embeddings_normalized = q_embeddings / q_norms
    else:
        print("Normalizing provided query embeddings...")
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1
        q_embeddings_normalized = query_embeddings / q_norms

    results = []
    for pair in pairs_to_check:
        idx1 = response_to_idx[pair["bad_response"]]
        idx2 = response_to_idx[pair["neighbor_response"]]

        resp_sim = float(
            np.dot(resp_embeddings_normalized[idx1], resp_embeddings_normalized[idx2])
        )

        if resp_sim <= response_similarity_threshold:
            continue

        if query_embeddings is None:
            q_idx1 = query_to_idx[pair["bad_query"]]
            q_idx2 = query_to_idx[pair["neighbor_query"]]
            prompt_sim = float(
                np.dot(q_embeddings_normalized[q_idx1], q_embeddings_normalized[q_idx2])
            )
        else:
            q_idx1 = pair["bad_query_index"]
            q_idx2 = pair["neighbor_query_index"]

            if (
                q_idx1 is None
                or q_idx2 is None
                or q_idx1 >= len(q_embeddings_normalized)
                or q_idx2 >= len(q_embeddings_normalized)
            ):
                prompt_sim = 0.0
            else:
                prompt_sim = float(
                    np.dot(
                        q_embeddings_normalized[q_idx1], q_embeddings_normalized[q_idx2]
                    )
                )

        if prompt_sim > prompt_similarity_threshold:
            result_item = pair.copy()
            result_item["response_similarity"] = resp_sim
            result_item["prompt_similarity"] = prompt_sim
            results.append(result_item)

    results.sort(key=lambda x: x["response_similarity"], reverse=True)

    return results

def generate_llm_prompts_for_contradictions(
    graph: QueryInfluenceGraph,
    mutual_contradictions: Dict[str, List[Dict[str, Any]]],
    task_guidelines: str = "Provide accurate, helpful, and clear responses.",
    data_review_instructions: List[str] = _INSTRUCTION,
    skip_identical_responses: bool = True,
    structured_output: bool = True,
    include_proposed_response: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generates LLM evaluation prompts for all unique pairs in mutual_contradictions.
    Returns a list of prompt dictionaries for evaluate_contradictions_with_llm.
    """
    prompts = []
    processed_pairs = set()

    for query, conflicts in mutual_contradictions.items():
        for conflict in conflicts:
            query_response = conflict["query_response"]
            neighbor = conflict["example_query"]
            neighbor_response = conflict["example_response"]

            pair_key = tuple(
                sorted([(query, query_response), (neighbor, neighbor_response)])
            )
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            if skip_identical_responses and query_response == neighbor_response:
                continue

            instructions = data_review_instructions.copy()

            if structured_output:
                instructions.append("")
                instructions.append(
                    "You must output exclusively in the following JSON format without Markdown wrapping:"
                )
                instructions.append("{")
                instructions.append('  "is_contradiction": true,')
                instructions.append('  "reasoning": "Explanation here",')
                instructions.append(
                    '  "better_record": "1"  // use "1", "2", or "Neither",'
                )
                if include_proposed_response:
                    instructions.append(
                        '  "proposed_new_response": "..." // for the worse record'
                    )
                instructions.append("}")

            instructions.append("")
            instructions.append(f"Guidelines:\n{task_guidelines}")

            system_instruction = "\n".join(instructions)

            user_prompt = (
                f"Record 1:\nInput: {query}\nTarget Response: {query_response}\n\n"
                f"Record 2:\nInput: {neighbor}\nTarget Response: {neighbor_response}\n\n"
                "Are they contradictory? Which record is better?"
            )

            query_node = graph._get_node(query, query_response)
            neighbor_node = graph._get_node(neighbor, neighbor_response)

            prompts.append(
                {
                    "system_instruction": system_instruction,
                    "user_prompt": user_prompt,
                    "record_a_index": query_node.query_index if query_node else None,
                    "record_a_query": query,
                    "record_a_response": query_response,
                    "record_b_index": neighbor_node.query_index
                    if neighbor_node
                    else None,
                    "record_b_query": neighbor,
                    "record_b_response": neighbor_response,
                }
            )

    return prompts

def generate_llm_prompts_for_bad_neighbors(
    bad_neighbors_pairs: List[Dict[str, Any]],
    task_guidelines: str = "Provide accurate, helpful, and clear responses.",
    data_review_instructions: List[str] = _INSTRUCTION,
    skip_identical_responses: bool = True,
    structured_output: bool = True,
    include_proposed_response: bool = True,
    record_names: Tuple[str, str] = ("Record 1", "Record 2"),
    score_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Generates LLM evaluation prompts for pairs found by find_similar_bad_neighbors.
    Returns a list of prompt dictionaries for evaluate_contradictions_with_llm.

    Args:
      record_names: Labels for the flagged record and its neighbor in the user
        prompt; they must match the labels used in `data_review_instructions`.
      score_name: If set (e.g. "Human Helpfulness Score"), each record in the
        user prompt also shows its external score (`bad_score` /
        `neighbor_score` from find_similar_bad_neighbors) under this name.
      structured_output: Appends a generic JSON output schema. Set to False when
        `data_review_instructions` already define their own output format.
    """
    prompts = []
    processed_pairs = set()

    for pair in bad_neighbors_pairs:
        bad_query = pair["bad_query"]
        bad_response = pair["bad_response"]
        bad_index = pair.get("bad_query_index")

        neighbor_query = pair["neighbor_query"]
        neighbor_response = pair["neighbor_response"]
        neighbor_index = pair.get("neighbor_query_index")

        pair_key = tuple(
            sorted([(bad_query, bad_response), (neighbor_query, neighbor_response)])
        )
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        if skip_identical_responses and bad_response == neighbor_response:
            continue

        instructions = data_review_instructions.copy()

        if structured_output:
            instructions.append("")
            instructions.append(
                "You must output exclusively in the following JSON format without Markdown wrapping:"
            )
            instructions.append("{")
            instructions.append('  "is_contradiction": true,')
            instructions.append('  "reasoning": "Explanation here",')
            instructions.append(
                '  "Verdict": "Neither"  // use "Annotator A Fault", "Annotator B Fault", "Valid Disagreement" or "Neither",'
            )
            if include_proposed_response:
                instructions.append(
                    '  "proposed_new_response": "..." // for the worse record'
                )
            instructions.append("}")

        instructions.append("")
        instructions.append(f"Guidelines:\n{task_guidelines}")

        def _record_block(name, query, response, score):
            block = f"{name}:\nInput: {query}\nTarget Response: {response}\n"
            if score_name is not None:
                block += f"{score_name}: {score}\n"
            return block

        user_prompt = (
            _record_block(record_names[0], bad_query, bad_response, pair.get("bad_score"))
            + "\n"
            + _record_block(
                record_names[1], neighbor_query, neighbor_response, pair.get("neighbor_score")
            )
            + "\nAre they contradictory? Which record is better?"
        )

        prompts.append(
            {
                "system_instruction": "\n".join(instructions),
                "user_prompt": user_prompt,
                "record_a_index": bad_index,
                "record_a_query": bad_query,
                "record_a_response": bad_response,
                "record_b_index": neighbor_index,
                "record_b_query": neighbor_query,
                "record_b_response": neighbor_response,
            }
        )

    return prompts

def find_high_impact_records(
    graph: QueryInfluenceGraph,
    advantage_threshold: float = -2.0,
    positively_impact: bool = False,
    min_students: int = 0,
) -> List[Dict[str, Any]]:
    """
    Finds individual records that positively/negatively impact the model's performance on neighboring queries,
    measured by comparing their exemplifying advantage against the provided threshold (Z-score).

    Args:
      graph: The scored graph.
      advantage_threshold: Z-score cut-off on `exemplifying_advantage`.
      positively_impact: If True, return records above the threshold instead of below.
      min_students: Minimum out-degree (number of exemplifying students) required
        for a record to be flagged. Default 0 applies no filter.
    """
    high_impact_records = []

    for query, query_dict in graph.nodes.items():
        for response, node in query_dict.items():
            q_ing_adv = node.exemplifying_advantage
            if q_ing_adv is None:
                continue
            if len(node.exemplifying_scores) < min_students:
                continue
            if positively_impact and q_ing_adv < advantage_threshold:
                continue
            if not positively_impact and q_ing_adv > advantage_threshold:
                continue
            high_impact_records.append(
                {
                    "query": query,
                    "response": response,
                    "query_index": node.query_index,
                    "exemplifying_advantage": q_ing_adv,
                    "num_students": len(node.exemplifying_scores),
                }
            )

    return high_impact_records

def find_mutual_NN_contradictions(
    graph: QueryInfluenceGraph,
    top_k: int = 15,
    advantage_threshold: float = -2.0,
    min_exemplified_contradictions: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Finds pairs of queries that are mutual nearest neighbors (in each other's top_k)
    where one of them degrades the other's score.

    Only considers a query for mutual contradictions if its `exemplified_contradiction`
    count exceeds `min_exemplified_contradictions` (meaning it has enough neighbor
    examples that degrade the query's score).
    """
    mutual_contradictions = collections.defaultdict(list)
    exemplified_contradiction_counts = collections.defaultdict(int)

    for query, query_dict in graph.nodes.items():
        for response, node in query_dict.items():
            base_score = node.base_score
            exemplified_scores = node.exemplified_scores

            q_ex_adv = node.exemplified_advantage
            q_ing_adv = node.exemplifying_advantage
            poor_advantage = (
                q_ex_adv is not None and q_ex_adv < advantage_threshold
            ) or (q_ing_adv is not None and q_ing_adv < advantage_threshold)

            if not exemplified_scores or base_score is None:
                continue

            for example in exemplified_scores[:top_k]:
                if example.score < base_score and poor_advantage:
                    exemplified_contradiction_counts[(query, response)] += 1

    for query, query_dict in graph.nodes.items():
        for response, node in query_dict.items():
            if (
                exemplified_contradiction_counts[(query, response)]
                <= min_exemplified_contradictions
            ):
                continue

            base_score = node.base_score
            exemplified_scores = node.exemplified_scores

            if not exemplified_scores or base_score is None:
                continue

            for example in exemplified_scores[:top_k]:
                score_diff = example.score - base_score
                if score_diff >= 0:
                    continue

                q_ex_adv = node.exemplified_advantage
                q_ing_adv = node.exemplifying_advantage

                poor_advantage = False
                if (q_ex_adv is not None and q_ex_adv < advantage_threshold) or (
                    q_ing_adv is not None and q_ing_adv < advantage_threshold
                ):
                    poor_advantage = True

                if not poor_advantage:
                    continue

                example_node = graph._get_node(example.query, example.response)
                if not example_node:
                    continue

                example_neighbors = example_node.exemplified_scores
                if not example_neighbors:
                    continue

                is_mutual = any(
                    n.query == query and n.response == response
                    for n in example_neighbors[:top_k]
                )

                if is_mutual:
                    mutual_contradictions[query].append(
                        {
                            "query_response": response,
                            "example_query": example.query,
                            "example_response": example.response,
                            "score_diff": score_diff,
                        }
                    )

    return dict(mutual_contradictions)

# Curation rules (see curation_rules.py) that can rank Stage-1 candidates.
RULE_RANKED_STAGE1 = ("calibrated_margin", "exemplifying_advantage")


def rank_records_by_rule(
    graph: QueryInfluenceGraph,
    rule: Any,
    budget: int | None = None,
    min_students: int = 0,
) -> List[Dict[str, Any]]:
    """Returns the records a curation rule ranks lowest (most suspicious) first.

    Uses `curation_rules.curate_graph` to compute each record's `rank_score`
    (higher = more prototypical for `calibrated_margin`, more helpful for
    `exemplifying_advantage`) and returns records in ascending score order. Records
    without a finite score are skipped. The rule's uniqueness guard is not applied:
    selected records are only sent to arbitration, never deleted here.

    Args:
      graph: The scored graph.
      rule: A registered rule name or a `curation_rules.FilteringRule` instance.
      budget: Maximum number of records to return (None = all).
      min_students: Minimum out-degree (number of exemplifying students).

    Returns:
      Dicts with `query`, `response`, `query_index`, `rule_score` and the node's
      `exemplifying_advantage` / `num_students`, most suspicious first.
    """
    from curation_rules import curate_graph

    result = curate_graph(graph, rule=rule, target_percentage=1.0)
    ranked = []
    for query, query_dict in graph.nodes.items():
        for response, node in query_dict.items():
            score = result.rank_scores.get((query, response))
            if score is None or not math.isfinite(score):
                continue
            if len(node.exemplifying_scores) < min_students:
                continue
            ranked.append(
                {
                    "query": query,
                    "response": response,
                    "query_index": node.query_index,
                    "rule_score": float(score),
                    "exemplifying_advantage": node.exemplifying_advantage,
                    "num_students": len(node.exemplifying_scores),
                }
            )
    ranked.sort(key=lambda r: (r["rule_score"], r["query_index"]))
    return ranked if budget is None else ranked[:budget]


def find_worst_neighbor_twins(
    graph: QueryInfluenceGraph,
    budget: int = 8500,
    top_k: int = 15,
    anchor_rule: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ranks all records by |Adv_e(n_i) - median(Adv_e)| and pairs each top-ranked
    anchor n_i with its single most-degrading incoming neighbor:
        n_{j*(i)} = argmin_{n_j in N_in(n_i)} Delta_LL_{i,j}

    If `anchor_rule` (a registered rule name or `FilteringRule`) is given, anchors
    are instead the `budget` records with the lowest rule `rank_score`
    (see `rank_records_by_rule`), paired the same way.
    """
    import numpy as np

    if anchor_rule is not None:
        scored_anchors = []
        for rec in rank_records_by_rule(graph, anchor_rule):
            node = graph._get_node(rec["query"], rec["response"])
            if node is None or not node.exemplified_scores:
                continue
            # Higher sort key = more suspicious, matching the |Adv_e| ordering below.
            scored_anchors.append(
                (-rec["rule_score"], node.query_index or 0, rec["query"], rec["response"], node)
            )
    else:
        nodes_with_adv = []
        for query, query_dict in graph.nodes.items():
            for response, node in query_dict.items():
                if node.exemplified_advantage is not None and node.exemplified_scores:
                    nodes_with_adv.append((query, response, node))

        if not nodes_with_adv:
            return {}

        adv_e_vals = np.array([n.exemplified_advantage for _, _, n in nodes_with_adv], dtype=float)
        med = float(np.median(adv_e_vals))

        scored_anchors = []
        for query, response, node in nodes_with_adv:
            dev = abs(float(node.exemplified_advantage) - med)
            scored_anchors.append((dev, node.query_index or 0, query, response, node))

    scored_anchors.sort(key=lambda x: (-x[0], x[1]))
    selected = scored_anchors[:budget]

    twin_pairs = collections.defaultdict(list)
    for dev, _, query, response, node in selected:
        candidates = [ex for ex in node.exemplified_scores[:top_k] if ex.query != query]
        if not candidates:
            continue
        worst_ex = min(candidates, key=lambda ex: ex.score)
        base_score = node.base_score if node.base_score is not None else 0.0
        pair = {
            "query_response": response,
            "example_query": worst_ex.query,
            "example_response": worst_ex.response,
            "score_diff": worst_ex.score - base_score,
        }
        if anchor_rule is None:
            pair["abs_adv_e_deviation"] = dev
        else:
            pair["rule_score"] = -dev
        twin_pairs[query].append(pair)

    return dict(twin_pairs)
