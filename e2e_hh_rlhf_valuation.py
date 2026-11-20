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

import os
import json
import argparse
import random
import hashlib
import litellm

from valuation_pipeline import DataValuationPipeline
from graph_analysis import (
    RULE_RANKED_STAGE1,
    find_mutual_NN_contradictions,
    find_worst_neighbor_twins,
    rank_records_by_rule,
    generate_llm_prompts_for_contradictions,
    evaluate_contradictions_with_llm,
)

def parse_hh_rlhf_row(row):
    chosen = str(row["chosen"])
    rejected = str(row["rejected"])

    split_token = "\n\nAssistant: "
    last_idx = chosen.rfind(split_token)

    if last_idx == -1:
        return chosen, chosen, rejected

    shared_context = chosen[: last_idx + len(split_token)]
    chosen_resp = chosen[last_idx + len(split_token) :]

    last_idx_rej = rejected.rfind(split_token)
    if last_idx_rej != -1:
        rejected_resp = rejected[last_idx_rej + len(split_token) :]
    else:
        rejected_resp = rejected

    return shared_context, chosen_resp, rejected_resp

def get_shuffled_hh_rlhf_options(shared_context, chosen_resp, rejected_resp):
    options = [(chosen_resp, True), (rejected_resp, False)]
    seed = int(hashlib.md5(shared_context.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(options)
    return options

def hh_rlhf_query_formatter(row):
    shared_context, chosen_resp, rejected_resp = parse_hh_rlhf_row(row)
    options = get_shuffled_hh_rlhf_options(shared_context, chosen_resp, rejected_resp)

    prompt_parts = [
        "Given the following conversation, evaluate the two options for the Assistant's next response and select the best one based on the Helpful, Harmless, and Honest (HHH) criteria.",
        "The best response should be as helpful and informative as possible, while strictly avoiding any harmful, toxic, or dangerous content.\n",
        "Conversation:",
        shared_context.strip(),
        "\n",
    ]

    for i, (text, _) in enumerate(options):
        prompt_parts.append(f"Option {i + 1}:\n{text.strip()}\n")

    prompt_parts.append("Which is the better response? (Answer 1 or 2)")
    prompt_parts.append("Answer:")
    return "\n".join(prompt_parts)

def hh_rlhf_response_formatter(row):
    shared_context, chosen_resp, rejected_resp = parse_hh_rlhf_row(row)
    options = get_shuffled_hh_rlhf_options(shared_context, chosen_resp, rejected_resp)

    for i, (_, is_chosen) in enumerate(options):
        if is_chosen:
            return f" {i + 1}"
    return " 1"

def hh_rlhf_label_pair_formatter(row):
    """Observed label and the other label, for margin scoring (`--score_both_labels`).

    With `score_difference=True` the pipeline scores both completions and uses
    log p(observed label) - log p(other label) as each record's score, and each
    neighbor is shown with its observed label.
    """
    observed = hh_rlhf_response_formatter(row)
    other = " 2" if observed == " 1" else " 1"
    return json.dumps({"chosen": observed, "rejected": other})

def hh_rlhf_embedding_formatter(row):
    shared_context, _, _ = parse_hh_rlhf_row(row)
    return shared_context.strip()

def _name_slug(name: str) -> str:
    """Turns a model name such as 'Qwen/Qwen3.5-9B' into 'qwen3.5-9b' for file names."""
    return name.rstrip("/").split("/")[-1].lower().replace(" ", "_")

_RLHF_INSTRUCTION = [
    "You are an expert Data Quality Evaluator. You are reviewing preference data (RLHF) records to ensure label consistency and truthfulness.",
    "",
    "CONTRADICTION DEFINITION:",
    "A pair is a REAL CONTRADICTION only if one record has an objectively 'flipped' label—meaning the [Target Response] (chosen option) is fundamentally inferior to the rejected option based on [Guidelines].",
    "",
    "GENERAL EVALUATION PRINCIPLE: FUNCTIONAL PARITY",
    "Prioritize the judgment (the choice between options) over the format. Do not excuse a contradiction just because one conversation has more turns or a different structure. If the core intent is similar, the resolution must be consistent.",
    "",
    "**Step-by-Step Instructions:**",
    "1. **Determine Core Intent:** Identify the user's primary goal in the [Input].",
    "2. **Independent Label Audit:** For each record, independently assess if the [Target Response] (the chosen option) is truly better than the alternative. Use the [Guidelines] (e.g., Harmlessness > Helpfulness).",
    "3. **Detect Judgment Conflict:** Compare the decisions. If Record 1 favors behavior X over Y, but Record 2 favors Y over X in a functionally similar context, a contradiction exists IF one of those choices is objectively wrong.",
    "4. **Threshold for Certainty:** If both options are poor, or if the choice is a 'toss-up' based on subjective preference, do NOT mark it as a contradiction. Only flag cases where a label should strictly be flipped to maintain dataset integrity.",
    "",
    "* CRITICAL OUTPUT:",
    "- 'is_contradiction': Set to true ONLY if at least one record's label is objectively incorrect.",
    "- 'better_record': Identify which record (Record 1 or Record 2) followed the guidelines correctly. If both are wrong or both are equally acceptable, put 'Neither'.",
    "- 'proposed_new_response': If a label should be flipped, explain which response index should have been chosen and why.",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HH-RLHF Data Valuation Pipeline")
    parser.add_argument(
        "--config", type=str, default="default.json", help="Path to JSON config file"
    )
    parser.add_argument(
        "--model_str",
        type=str,
        default=None,
        help="Model string used for output file names",
    )
    parser.add_argument(
        "--embedding_model", type=str, default=None, help="Embedding model name"
    )
    parser.add_argument(
        "--scoring_model", type=str, default=None, help="Scoring model name for vLLM"
    )
    parser.add_argument(
        "--prompt_logprobs",
        type=int,
        default=None,
        help="Number of logprobs to return per token",
    )
    parser.add_argument(
        "--embedding_ver_str", type=str, default="full", help="Embedding version string"
    )
    parser.add_argument(
        "--advantage_threshold",
        type=float,
        default=-2.0,
        help=(
            "Only used with --pairing_mode mutual_nn: a record is a candidate if its "
            "exemplified OR exemplifying advantage is below this Z-score."
        ),
    )
    parser.add_argument(
        "--eval_model", type=str, default=None, help="LLM model name for evaluation"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="hh_rlhf_evaluations.jsonl",
        help="Output path for LLM evaluations",
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default="hh_rlhf_contradictions_report.txt",
        help="Output path for contradiction report",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="LLM API key (or set the LLM_API_KEY environment variable).",
    )
    parser.add_argument(
        "--score_both_labels",
        action="store_true",
        help=(
            "Score both option labels and use the margin log p(observed label) - "
            "log p(other label) as the score (doubles scoring cost). Graph and "
            "cache file names get a '_margin' suffix."
        ),
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Only use the first N records of the split (for quick test runs).",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=16,
        help="Nearest-neighbor query size including the record itself "
        "(k_neighbors - 1 one-shot exemplars per record).",
    )
    parser.add_argument(
        "--data_split", type=str, default="train", help="Dataset split to use. Can be 'train' or 'test'."
    )
    parser.add_argument(
        "--pairing_mode",
        type=str,
        default="worst_neighbor",
        choices=["worst_neighbor", "mutual_nn"],
        help="Stage 1 pairing rule: 'worst_neighbor' (top-B anchors ranked by |Adv_e - median(Adv_e)|, each paired with its argmin incoming neighbor) or 'mutual_nn' (mutual opposing edges).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=8500,
        help="Candidate budget B for worst_neighbor pairing mode.",
    )
    parser.add_argument(
        "--stage1_rule",
        type=str,
        default="adv_e_deviation",
        choices=("adv_e_deviation",) + RULE_RANKED_STAGE1,
        help=(
            "How worst_neighbor mode picks its --budget anchors: 'adv_e_deviation' "
            "(default) ranks by |Adv_e - median(Adv_e)|; a curation rule name takes the "
            "records with the lowest rule score (see curation_rules.py). With a rule, "
            "the anchors and their scores are also written to --anchors_output, with "
            "or without --eval_model."
        ),
    )
    parser.add_argument(
        "--anchors_output",
        type=str,
        default=None,
        help="Output JSONL for rule-selected anchors (default: derived from the graph name).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Configuration file not found: {args.config}. Please provide a valid JSON config."
        )

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.model_str:
        config["model_str"] = args.model_str
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model
    if args.prompt_logprobs is not None:
        config["prompt_logprobs"] = args.prompt_logprobs
    if args.scoring_model:
        if "vllm_config" not in config:
            config["vllm_config"] = {}
        config["vllm_config"]["model"] = args.scoring_model

    eval_model = args.eval_model or config.get("eval_model")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")

    if eval_model:
        is_known_model = (
            hasattr(litellm, "model_list") and eval_model in litellm.model_list
        )
        has_valid_provider = (
            hasattr(litellm, "provider_list")
            and "/" in eval_model
            and eval_model.split("/")[0] in litellm.provider_list
        )

        if not (is_known_model or has_valid_provider):
            raise ValueError(
                f"Evaluation model '{eval_model}' is not in a recognized LiteLLM format. "
                "Please use a natively supported model or add a valid provider prefix (e.g., '<provider>/<model>')."
            )

    if args.stage1_rule != "adv_e_deviation" and args.pairing_mode != "worst_neighbor":
        raise ValueError("--stage1_rule requires --pairing_mode worst_neighbor.")

    if args.pairing_mode == "worst_neighbor" and args.advantage_threshold != -2.0:
        print(
            "Note: --advantage_threshold is ignored with --pairing_mode "
            "worst_neighbor (anchors are ranked by |Adv_e - median(Adv_e)|)."
        )

    _MODEL_STR = config.get("model_str") or _name_slug(config["vllm_config"]["model"])
    _EMB_STR = _name_slug(config["embedding_model"])
    _RUN_STR = args.embedding_ver_str
    if args.sample_size is not None:
        _RUN_STR += f"_n{args.sample_size}"
    _SCORE_STR = _RUN_STR + ("_margin" if args.score_both_labels else "")

    pipeline = DataValuationPipeline(
        query_formatting_fn=hh_rlhf_query_formatter,
        response_formatting_fn=(
            hh_rlhf_label_pair_formatter
            if args.score_both_labels
            else hh_rlhf_response_formatter
        ),
        score_difference=args.score_both_labels,
        embedding_formatting_fn=hh_rlhf_embedding_formatter,
        embedding_model_name=config["embedding_model"],
        prompt_logprobs=config.get("prompt_logprobs", 10),
        vllm_config=config.get("vllm_config"),
        k_neighbors=args.k_neighbors,
        output_embeddings_path=f"hh_rlhf_{args.data_split}_{_EMB_STR}_{_RUN_STR}_embeddings.npy",
        output_neighbors_path=f"hh_rlhf_{args.data_split}_{_EMB_STR}_{_RUN_STR}_neighbors.npz",
        output_scoring_inputs_path=f"hh_rlhf_{args.data_split}_{_MODEL_STR}_{_SCORE_STR}_scoring_inputs.jsonl",
        output_scored_records_path=f"hh_rlhf_{args.data_split}_{_MODEL_STR}_{_SCORE_STR}_scored_records.jsonl",
        output_graph_path=f"hh_rlhf_{args.data_split}_{_MODEL_STR}_{_SCORE_STR}_graph.jsonl",
        errors_output_path=f"hh_rlhf_{args.data_split}_{_MODEL_STR}_{_SCORE_STR}_errors.jsonl",
    )

    print("\n" + "=" * 50)
    print("Building graph for HH-RLHF")
    print("=" * 50)

    graph = pipeline.run(
        dataset_kwargs={
            "dataset_name": "Anthropic/hh-rlhf",
            "dataset_split": args.data_split,
        },
        sample_size=args.sample_size,
    )
    print(f"\nSuccessfully built graph with {len(graph.nodes)} unique queries.")

    anchor_rule = None if args.stage1_rule == "adv_e_deviation" else args.stage1_rule
    if anchor_rule is not None:
        anchors_output = args.anchors_output or (
            f"hh_rlhf_{args.data_split}_{_MODEL_STR}_{_SCORE_STR}_{anchor_rule}_anchors.jsonl"
        )
        anchors = rank_records_by_rule(graph, anchor_rule, budget=args.budget)
        with open(anchors_output, "w", encoding="utf-8") as f:
            for rec in anchors:
                f.write(json.dumps({"rule": anchor_rule, **rec}) + "\n")
        print(f"Wrote {len(anchors)} Stage-1 anchors ({anchor_rule}) to {anchors_output}")

    if eval_model:
        print("\n" + "=" * 50)
        print("Graph Analysis & Contradiction Evaluation")
        print("=" * 50)

        if args.pairing_mode == "worst_neighbor":
            candidate_pairs = find_worst_neighbor_twins(
                graph, budget=args.budget, anchor_rule=anchor_rule
            )
        else:
            candidate_pairs = find_mutual_NN_contradictions(
                graph, advantage_threshold=args.advantage_threshold
            )
        print(
            f"Found {sum(len(v) for v in candidate_pairs.values())} candidate contradiction pairs."
        )

        if candidate_pairs:
            prompts = generate_llm_prompts_for_contradictions(
                graph,
                candidate_pairs,
                task_guidelines="Provide accurate, helpful, and clear responses.",
                data_review_instructions=_RLHF_INSTRUCTION,
                include_proposed_response=True,
                # HH-RLHF targets are option labels (" 1"/" 2"), so twins that
                # chose the same label must still be arbitrated.
                skip_identical_responses=False,
            )
            print(f"Generated {len(prompts)} prompts to be sent to LLM for review.")

            evaluated = evaluate_contradictions_with_llm(
                prompts,
                model_name=eval_model,
                api_key=api_key,
                temperature=0,
                max_workers=10,
                output_path=args.eval_output,
            )

            real_contradiction = []
            records_with_issue = set()
            for idx, result in enumerate(evaluated):
                model_eval = result.get("llm_evaluation", {})
                if isinstance(model_eval, list) and len(model_eval) == 1:
                    model_eval = model_eval[0]

                if not model_eval:
                    continue

                if model_eval.get("is_contradiction") == True:
                    real_contradiction.append(result)

                    better_record = (
                        str(model_eval.get("better_record", "")).strip().lower()
                    )
                    if better_record == "1":
                        if result.get("record_b_index") is not None:
                            records_with_issue.add(result["record_b_index"])
                    elif better_record == "2":
                        if result.get("record_a_index") is not None:
                            records_with_issue.add(result["record_a_index"])
                    elif better_record == "neither":
                        if result.get("record_a_index") is not None:
                            records_with_issue.add(result["record_a_index"])
                        if result.get("record_b_index") is not None:
                            records_with_issue.add(result["record_b_index"])

            print(f"Number of real contradiction: {len(real_contradiction)}")
            print(f"Number of records with issue: {len(records_with_issue)}")
            print("\nWriting evaluation results to file...")
            with open(args.report_output, "w", encoding="utf-8") as f_out:
                for idx, result in enumerate(real_contradiction):
                    model_eval = result.get("llm_evaluation", {})
                    if isinstance(model_eval, list) and len(model_eval) == 1:
                        model_eval = model_eval[0]

                    f_out.write(f"\n-------- Contradiction Pair {idx + 1} --------\n")
                    f_out.write(f"Record 1 (Index {result['record_a_index']}):\n")
                    f_out.write(
                        f"Query: {result['record_a_query']} \nTarget Response: {result['record_a_response']}\n\n"
                    )
                    f_out.write(f"Record 2 (Index {result['record_b_index']}):\n")
                    f_out.write(
                        f"Query: {result['record_b_query']} \nTarget Response: {result['record_b_response']}\n\n"
                    )
                    f_out.write(f"LLM Verdict:\n")
                    f_out.write(
                        f"Is Contradiction: {model_eval.get('is_contradiction')}\n"
                    )
                    f_out.write(f"Better Record: {model_eval.get('better_record')}\n")
                    f_out.write(f"Reasoning: {model_eval.get('reasoning')}\n")
                    f_out.write(
                        f"Proposed New Response: {model_eval.get('proposed_new_response')}\n"
                    )
            print(f"\nSaved formatted contradiction report to {args.report_output}")
        else:
            print("No candidate pairs found, skipping evaluation.")
