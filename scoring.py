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

import json
import os
import time
import math
from typing import List, Optional, Sequence, Tuple, Dict, Any
from tqdm import tqdm


os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams


def validate_prompt_logic(
    record, query_text, query_index, neighbor_text=None, neighbor_response=None
):
    """Checks that a record produced by `create_vllm_inputs` is self-consistent."""
    try:
        prompt = record["prompt"]
        if record["query_text"] != query_text:
            print(f"Mismatch Query: {record['query_text']} != {query_text}")
            return False
        if record["query_index"] != query_index:
            print(f"Mismatch Index: {record['query_index']} != {query_index}")
            return False
        if not prompt.endswith(query_text):
            print("Prompt does not end with the query text")
            return False

        if record.get("example_query") is not None:
            if record["example_query"] != neighbor_text:
                print(f"Mismatch ExQuery: {record['example_query']} != {neighbor_text}")
                return False
            if record["example_response"] != neighbor_response:
                print(
                    f"Mismatch ExResponse: {record['example_response']} != {neighbor_response}"
                )
                return False
            exemplar = f"<question>{neighbor_text}</question><answer>{neighbor_response}</answer>"
            if exemplar not in prompt:
                print("Exemplar block not found in prompt")
                return False
        return True
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

ONE_SHOT_INSTRUCTION = (
    "Answer the question considering the example question and answer pair. "
)

def create_vllm_inputs(
    query_text,
    target_response,
    neighbors,
    global_id,
    general_instruction="",
    writer=None,
):
    """Builds the 0-shot and 1-shot scoring records for one query.

    Prompt formats (`{TQ}` is the target query, `{NQ}`/`{NA}` a neighbor's
    query and response):
      0-shot: {general_instruction}{TQ}
      1-shot: {general_instruction}{ONE_SHOT_INSTRUCTION}<question>{NQ}</question><answer>{NA}</answer>{TQ}
    The dataset row index is never placed in the prompt; it is carried only as
    record metadata.

    Record fields:
      prompt: Text presented to the model before the scored completion.
      scoring_completions: The completion whose log-likelihood is measured.
      query_text: The formatted query, verbatim.
      query_index: `global_id`, the dataset row index of the query.
      example_query / example_response: The exemplar for 1-shot records, None
        for the base record.
    """
    records = []

    base_record = {
        "prompt": f"{general_instruction}{query_text}",
        "scoring_completions": target_response,
        "query_text": query_text,
        "query_index": global_id,
        "example_query": None,
        "example_response": None,
    }
    records.append(base_record)
    if writer:
        writer.write(json.dumps(base_record) + "\n")

    for n_text, n_resp in neighbors:
        prompt = (
            f"{general_instruction}{ONE_SHOT_INSTRUCTION}"
            f"<question>{n_text}</question><answer>{n_resp}</answer>{query_text}"
        )
        record = {
            "prompt": prompt,
            "scoring_completions": target_response,
            "query_text": query_text,
            "query_index": global_id,
            "example_query": n_text,
            "example_response": n_resp,
        }
        records.append(record)
        if writer:
            writer.write(json.dumps(record) + "\n")

    return records

def read_vllm_inputs(file_path: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file containing vLLM inputs back into a list of dictionaries."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def calculate_batch_log_likelihoods(
    llm: LLM,
    tokenizer: Any,
    prompts: List[str],
    completions: List[str],
    prompt_logprobs: int = 10,
    return_num_tokens: bool = False,
    return_truncated: bool = False,
):
    """Calculates log-likelihoods for a batch of prompt and completion pairs.

    The returned value for each pair is the total log-likelihood summed over
    the completion tokens. Pass `return_num_tokens=True` to also get the number
    of completion tokens that were scored, so callers can normalise by length,
    and `return_truncated=True` to also get, per pair, whether the prompt or
    completion had to be shortened to fit the model context (prompts are cut
    from the left, so a truncated 1-shot prompt loses its instruction and the
    start of its exemplar).
    """

    max_model_len = llm.llm_engine.model_config.max_model_len

    full_token_ids_batch = []

    all_p_toks = tokenizer(prompts, add_special_tokens=True)["input_ids"]
    all_c_toks = tokenizer(completions, add_special_tokens=False)["input_ids"]

    full_token_ids_batch = []
    actual_p_lens = []
    actual_c_lens = []
    truncated = []

    for p_toks, c_toks in zip(all_p_toks, all_c_toks):
        was_truncated = False
        if len(c_toks) >= max_model_len - 2:
            c_toks = c_toks[: max_model_len - 5]
            was_truncated = True

        allowed_p_len = max(1, max_model_len - len(c_toks) - 1)

        if len(p_toks) > allowed_p_len:
            # Keep the first token (the BOS token for tokenizers that add one)
            # and the end of the prompt, which holds the target query.
            p_toks = [p_toks[0]] + p_toks[-(allowed_p_len - 1) :]
            was_truncated = True

        actual_p_lens.append(len(p_toks))
        actual_c_lens.append(len(c_toks))
        truncated.append(was_truncated)
        full_token_ids_batch.append(p_toks + c_toks)

    if any(truncated):
        print(
            f"WARNING: {sum(truncated)} of {len(truncated)} sequences exceeded the "
            f"model context ({max_model_len} tokens) and were truncated."
        )

    sampling_params = SamplingParams(
        max_tokens=1, prompt_logprobs=prompt_logprobs, detokenize=False
    )

    print(
        f"Calculating log-likelihoods for {len(prompts)} sequences (Max Len: {max_model_len})..."
    )
    vllm_inputs = [{"prompt_token_ids": ids} for ids in full_token_ids_batch]
    outputs = llm.generate(
        prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False
    )

    results = []
    vocab_size = len(tokenizer)
    for i, output in enumerate(outputs):
        p_len = actual_p_lens[i]
        total_logprob = 0.0

        for j in range(p_len, len(output.prompt_token_ids)):
            token_logprobs = output.prompt_logprobs[j]

            if token_logprobs is None:
                total_logprob += -15.0  # Fallback penalty for missing logprobs at this position (e.g., first token j=0)
                continue

            target_id = output.prompt_token_ids[j]

            if target_id in token_logprobs:
                lp_obj = token_logprobs[target_id]
                total_logprob += getattr(lp_obj, "logprob", lp_obj)
            else:
                top_k_sum = sum(
                    math.exp(getattr(lp, "logprob", lp))
                    for lp in token_logprobs.values()
                )
                remaining = max(1e-9, 1.0 - top_k_sum)
                avg_missing = remaining / max(1, vocab_size - len(token_logprobs))
                total_logprob += math.log(avg_missing)

        results.append(total_logprob)
    extras = []
    if return_num_tokens:
        extras.append(actual_c_lens)
    if return_truncated:
        extras.append(truncated)
    return (results, *extras) if extras else results

def first_occurrence_rows(queries: Sequence[str], responses: Sequence[str]) -> List[int]:
    """Returns the row positions of the first occurrence of each (query, response) pair.

    Graph nodes are keyed by (query, response) text, so exact duplicate rows would
    collapse into one node and could be used as each other's exemplar (a self-loop).
    The pipeline therefore scores only the first occurrence of each pair.
    """
    seen = set()
    rows = []
    for i, key in enumerate(zip(queries, responses)):
        key = (str(key[0]), str(key[1]))
        if key not in seen:
            seen.add(key)
            rows.append(i)
    return rows

def _select_neighbor_indices(
    neighbor_row, query_index: int, max_neighbors: int
) -> List[int]:
    """Selects neighbor indices for a query, excluding the query itself by identity.

    Negative values mark padding (no neighbor) and are skipped.
    """
    selected = []
    for value in neighbor_row:
        neighbor_idx = int(value)
        if neighbor_idx < 0 or neighbor_idx == query_index:
            continue
        selected.append(neighbor_idx)
        if len(selected) >= max_neighbors:
            break
    return selected

def generate_dataset_inputs(
    dataset, labels, text_col: str, answer_col: str, general_instruction: str = "",
    score_difference: bool = False, query_rows: Optional[Sequence[int]] = None
) -> List[Dict[str, Any]]:
    """Takes a dataset and nearest-neighbor labels, and generates a list of input records for vLLM.

    Args:
      labels: Neighbor row indices per dataset row (negative values are padding).
      query_rows: Rows to generate records for (default: all rows). The pipeline
        passes the first occurrence of each (query, response) pair here.
    """
    all_records = []

    print("Loading dataset columns into memory...")
    texts = dataset[text_col]
    answers = dataset[answer_col]
    rows = range(len(dataset)) if query_rows is None else query_rows

    for i in tqdm(rows, desc="Generating dataset inputs"):
        i = int(i)
        query_text = str(texts[i])

        if score_difference:
            ans_data = json.loads(str(answers[i]))
            chosen_ans = ans_data["chosen"]
            rejected_ans = ans_data["rejected"]

            # Exclude the query itself by identity; see _select_neighbor_indices.
            # We use the CHOSEN response of the neighbor as the example.
            neighbor_indices = _select_neighbor_indices(
                labels[i], i, max_neighbors=len(labels[i]) - 1
            )
            neighbors = []
            for neighbor_idx in neighbor_indices:
                n_text = str(texts[neighbor_idx])
                n_ans_data = json.loads(str(answers[neighbor_idx]))
                n_chosen_ans = n_ans_data["chosen"]
                neighbors.append((n_text, n_chosen_ans))

            chosen_records = create_vllm_inputs(
                query_text, chosen_ans, neighbors, i, general_instruction
            )
            rejected_records = create_vllm_inputs(
                query_text, rejected_ans, neighbors, i, general_instruction
            )

            for r_idx, (c_rec, r_rec) in enumerate(zip(chosen_records, rejected_records)):
                # records[0] is the 0-shot base record; records[r] for r > 0 is
                # exemplified by neighbor_indices[r - 1].
                neighbor_index = neighbor_indices[r_idx - 1] if r_idx > 0 else None

                c_rec["query_index"] = i
                c_rec["is_base"] = r_idx == 0
                c_rec["neighbor_index"] = neighbor_index
                c_rec["is_chosen"] = True

                r_rec["query_index"] = i
                r_rec["is_base"] = r_idx == 0
                r_rec["neighbor_index"] = neighbor_index
                r_rec["is_chosen"] = False

                all_records.append(c_rec)
                all_records.append(r_rec)
        else:
            target_response = str(answers[i])

            # Exclude the query itself by identity
            neighbor_indices = _select_neighbor_indices(
                labels[i], i, max_neighbors=len(labels[i]) - 1
            )
            neighbors = []
            for neighbor_idx in neighbor_indices:
                n_text = str(texts[neighbor_idx])
                n_resp = str(answers[neighbor_idx])
                neighbors.append((n_text, n_resp))

            records = create_vllm_inputs(
                query_text, target_response, neighbors, i, general_instruction
            )

            for r_idx, record in enumerate(records):
                # records[0] is the 0-shot base record; records[r] for r > 0 is
                # exemplified by neighbor_indices[r - 1].
                record["query_index"] = i
                record["is_base"] = r_idx == 0
                record["neighbor_index"] = (
                    neighbor_indices[r_idx - 1] if r_idx > 0 else None
                )
                all_records.append(record)

    return all_records

_RECORD_IDENTITY_FIELDS = (
    "query_index",
    "is_base",
    "neighbor_index",
    "is_chosen",
    "prompt",
    "scoring_completions",
)

def _record_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(record.get(k) for k in _RECORD_IDENTITY_FIELDS)

def _load_scored_prefix(output_path: str) -> List[Dict[str, Any]]:
    """Loads previously scored records, dropping a cut-off final line.

    If the file ends with an incomplete line (an interrupted write), the file is
    truncated to the last complete record so new records are appended cleanly.
    """
    records = []
    valid_bytes = 0
    with open(output_path, "rb") as f:
        for raw in f:
            if not raw.strip():
                valid_bytes += len(raw)
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                break
            valid_bytes += len(raw)
    if valid_bytes < os.path.getsize(output_path):
        print(
            f"Dropping an incomplete trailing record from {output_path} "
            f"(keeping {len(records)} complete records)."
        )
        with open(output_path, "r+b") as f:
            f.truncate(valid_bytes)
    return records

def score_inputs(
    llm: LLM,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    chunk_size: int = 20000,
    output_path: str = None,
    start_chunk: int = 0,
    end_chunk: int = None,
    prompt_logprobs: int = 10,
    llm_factory=None,
) -> List[Dict[str, Any]]:
    """Scores a prepared list of vLLM inputs and adds the logprobs back to the dictionary.

    Pass `llm`/`tokenizer`, or `llm=None` and an `llm_factory` returning
    `(llm, tokenizer)`; the factory is only called if records remain to be scored.

    Each scored record gets `logprob`, `probability_pct`,
    `completion_num_tokens` and `prompt_truncated`. If `output_path` already
    holds scored records, scoring resumes after them; the existing records must
    match the current inputs (same queries, neighbors, prompts and completions,
    in order); otherwise a ValueError is raised.
    """

    # Isolate the specific range of records based on the chunk boundaries (useful when running the scoring on different machines)
    start_idx = start_chunk * chunk_size
    end_idx = end_chunk * chunk_size if end_chunk is not None else len(records)
    target_records = records[start_idx:end_idx]

    scored_records = []
    resume_index = 0

    if output_path and os.path.exists(output_path):
        print(f"\nFound existing scored records at {output_path}. Loading progress...")
        scored_records = _load_scored_prefix(output_path)
        if len(scored_records) > len(target_records):
            raise ValueError(
                f"{output_path} holds {len(scored_records)} scored records but this "
                f"run has only {len(target_records)} inputs. It was produced by a "
                "different run; delete it or choose another output path."
            )
        for done, expected in zip(scored_records, target_records):
            if _record_identity(done) != _record_identity(expected):
                raise ValueError(
                    f"{output_path} does not match the current scoring inputs (first "
                    f"mismatch at query_index={expected.get('query_index')}, "
                    f"neighbor_index={expected.get('neighbor_index')}). It was "
                    "produced by a different run (e.g. another dataset, neighbor "
                    "set or prompt format); delete it or choose another output path."
                )

        resume_index = len(scored_records)
        print(
            f"Resuming from record {resume_index} out of {len(target_records)} in this chunk range...\n"
        )

    if resume_index < len(target_records) and llm is None:
        if llm_factory is None:
            raise ValueError("score_inputs needs `llm` or `llm_factory` to score records.")
        llm, tokenizer = llm_factory()

    total_chunks = (len(target_records) - resume_index + chunk_size - 1) // chunk_size
    chunks_processed = 0
    overall_start_time = time.time()

    for i in range(resume_index, len(target_records), chunk_size):
        chunk_start_time = time.time()
        chunk = target_records[i : i + chunk_size]
        current_chunk_num = chunks_processed + 1

        global_start = start_idx + i
        global_end = start_idx + i + len(chunk)
        print(
            f"\nProcessing vLLM chunk {current_chunk_num}/{total_chunks} (Global Records {global_start} to {global_end})..."
        )

        prompts = [r["prompt"] for r in chunk]
        completions = [r["scoring_completions"] for r in chunk]

        logprobs, num_tokens, truncated = calculate_batch_log_likelihoods(
            llm,
            tokenizer,
            prompts,
            completions,
            prompt_logprobs=prompt_logprobs,
            return_num_tokens=True,
            return_truncated=True,
        )

        for record, lp, n_tok, trunc in zip(chunk, logprobs, num_tokens, truncated):
            record["logprob"] = lp
            record["probability_pct"] = math.exp(lp) * 100
            record["completion_num_tokens"] = n_tok
            record["prompt_truncated"] = trunc

        scored_records.extend(chunk)

        if output_path:
            with open(
                output_path, "a" if i > 0 or resume_index > 0 else "w", encoding="utf-8"
            ) as f:
                for record in chunk:
                    f.write(json.dumps(record) + "\n")

        chunks_processed += 1
        chunk_time = time.time() - chunk_start_time
        elapsed_time = time.time() - overall_start_time
        avg_chunk_time = elapsed_time / chunks_processed
        remaining_chunks = total_chunks - chunks_processed
        eta_minutes = (remaining_chunks * avg_chunk_time) / 60

        print(
            f"-> Chunk {current_chunk_num} finished in {chunk_time:.2f}s. "
            f"ETA for remaining {remaining_chunks} chunks: {eta_minutes:.1f} minutes."
        )

    return scored_records
