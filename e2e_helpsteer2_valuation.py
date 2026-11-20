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
import litellm
from datasets import load_dataset
from valuation_pipeline import DataValuationPipeline
from graph_analysis import (
    RULE_RANKED_STAGE1,
    find_high_impact_records,
    rank_records_by_rule,
    find_similar_bad_neighbors,
    generate_llm_prompts_for_bad_neighbors,
    evaluate_contradictions_with_llm,
)

HELPSTEER_INSTRUCTION = [
    "You are an expert Data Quality Evaluator. You are reviewing human preference data to detect annotator contradictions.",
    "",
    "You will be provided with two paired records ([Record A] and [Record B]). Each record consists of an Input, an AI Response, and a Human Helpfulness Score (0-4).",
    "",
    "CRITICAL CONTEXT:",
    "- [Record A] ALWAYS represents the Highly Scored record.",
    "- [Record B] ALWAYS represents the Poorly Scored record.",
    "",
    "These records share keyword similarity, but they might be committing completely different errors. Your job is to objectively determine if they share a structural error, or if they are completely unrelated behaviors. Prompts with only format or structure differences, like single-turn vs. mulitple-turn, should not be considered as unrelated.",
    "",
    "**Step-by-Step Instructions:**",
    "1. **Identify the Shared Error:** Read both inputs and responses. Did both models fail in the exact same way? (e.g., both missed a requested format, both provided questions without answers).",
    "2. **Evaluate Annotator A (The High Score):** Based on the true quality of Record A, did Annotator A lazily miss the shared error and wrongly give Record A a high score?",
    "3. **Evaluate Annotator B (The Low Score):** Based on the true quality of Record B, did Annotator B correctly identify the error, making their low score justified, or did they unfairly penalize it?",
    "4. **Render a Verdict:** Is this a true annotator contradiction where one human missed a critical flaw that the other caught?",
    "",
    "You must output exclusively in the following JSON format without Markdown wrapping. Do not include quotes around the boolean value:",
    "{",
    '  "reasoning": "First, state the shared structural error (if any). Second, explicitly state if Annotator A and/or Annotator B misjudged their respective responses. Finally, conclude your logic.",',
    '  "is_contradiction": <true/false>, // true ONLY if this is a REAL CONTRADICTION (Annotator A, B, or Both Fault). Output false if it is a Valid Disagreement or Unrelated Pairs.',
    '  "verdict": "[SELECT EXACTLY ONE: Annotator A Fault | Annotator B Fault | Both Annotators at Fault | Valid Disagreement | Unrelated Pairs]"',
    "}",
    "",
    "**Verdict Categories:**",
    "- 'Annotator A Fault' (REAL CONTRADICTION): Response B's low score is justified. Response A is functionally equivalent and shares the exact same flaws, meaning Annotator A erroneously gave it a 4.",
    "- 'Annotator B Fault' (REAL CONTRADICTION): Response A's perfect score is justified. Response B is functionally equivalent and highly helpful, meaning Annotator B unfairly penalized it.",
    "- 'Both Annotators at Fault' (REAL CONTRADICTION): Both responses are functionally equivalent and their true quality is mediocre (e.g., a 3). Annotator A over-scored, AND Annotator B under-scored.",
    "- 'Valid Disagreement' (NOT A CONTRADICTION): The responses are merely semantically similar. A subtle but critical functional difference (e.g., a hallucinated fact or ignored instruction in B) completely justifies the massive score gap.",
    "- 'Unrelated Pairs' (NOT A CONTRADICTION): Despite one of the ratings is not accurate, the two pairs of prompts and responses are completely unrelated.",
]

_EXEMPLIFYING_ADV_THRESHOLD = -2

def helpsteer2_query_formatter(row):
    return f"Prompt: {row['prompt']}\nResponse:"

def helpsteer2_response_formatter(row):
    return f" {row['response']}"

def helpsteer2_embedding_formatter(row):
    return str(row["prompt"]).strip()

def _name_slug(name: str) -> str:
    """Turns a model name such as 'Qwen/Qwen3.5-9B' into 'qwen3.5-9b' for file names."""
    return name.rstrip("/").split("/")[-1].lower().replace(" ", "_")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelpSteer2 Data Valuation Pipeline")
    parser.add_argument(
        "--config", type=str, default="default.json", help="Path to JSON config file"
    )
    parser.add_argument(
        "--model_str",
        type=str,
        default=None,
        help="Model string. Used only for output file names",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding model name using huggingface format, e.g., 'Qwen/Qwen3-Embedding-4B'",
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
        "--advantage_threshold",
        type=float,
        default=_EXEMPLIFYING_ADV_THRESHOLD,
        help="Threshold for exemplifying advantage",
    )
    parser.add_argument(
        "--eval_model", type=str, default=None, help="LLM model name for evaluation"
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default="helpsteer2_evaluations.jsonl",
        help="Output path for LLM evaluations",
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default="helpsteer2_contradictions_report.txt",
        help="Output path for contradiction report",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="LLM API key (or set the LLM_API_KEY environment variable).",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=16,
        help="Nearest-neighbor query size including the record itself "
        "(k_neighbors - 1 one-shot exemplars per record).",
    )
    parser.add_argument(
        "--length_normalization",
        type=str,
        choices=["none", "tokens", "chars"],
        default="none",
        help=(
            "How per-edge log-likelihood differences are scaled before averaging "
            "into delta: 'none' = total nats, 'tokens' = per "
            "completion token, 'chars' = per character."
        ),
    )
    parser.add_argument(
        "--stage1_rule",
        type=str,
        default="adv_delta_threshold",
        choices=("adv_delta_threshold",) + RULE_RANKED_STAGE1,
        help=(
            "How Stage 1 flags high-scored records: 'adv_delta_threshold' (default) "
            "keeps records with Adv_delta <= --advantage_threshold; a curation rule "
            "name takes the --budget records with the lowest rule score (see "
            "curation_rules.py) and also writes them to --anchors_output, with or "
            "without --eval_model."
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Number of records to flag with --stage1_rule (required with a rule).",
    )
    parser.add_argument(
        "--anchors_output",
        type=str,
        default=None,
        help="Output JSONL for rule-selected records (default: derived from the rule name).",
    )
    parser.add_argument(
        "--min_students",
        type=int,
        default=0,
        help=(
            "Only flag records whose delta was averaged over at least this many "
            "students. 0 disables the filter."
        ),
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

    if args.stage1_rule != "adv_delta_threshold" and (args.budget is None or args.budget <= 0):
        raise ValueError("--stage1_rule requires a positive --budget.")

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

    print("\n" + "=" * 50)
    print("STEP 1: Building Global Graph (Full Dataset) without scoring")
    print("=" * 50)

    _MODEL_STR = config.get("model_str") or _name_slug(config["vllm_config"]["model"])
    _EMB_STR = _name_slug(config["embedding_model"])
    _EMBEDDING_VER_STR_FULL = "full"
    _DATASET_STR_FULL = "helpsteer2_no_score"

    global_pipeline = DataValuationPipeline(
        query_formatting_fn=helpsteer2_query_formatter,
        response_formatting_fn=helpsteer2_response_formatter,
        embedding_formatting_fn=helpsteer2_embedding_formatter,
        embedding_model_name=config["embedding_model"],
        prompt_logprobs=config.get("prompt_logprobs", 10),
        vllm_config=config.get("vllm_config"),
        k_neighbors=args.k_neighbors,
        output_embeddings_path=f"{_DATASET_STR_FULL}_{_EMB_STR}_{_EMBEDDING_VER_STR_FULL}_embeddings.npy",
        output_neighbors_path=f"{_DATASET_STR_FULL}_{_EMB_STR}_{_EMBEDDING_VER_STR_FULL}_neighbors.npz",
        output_scoring_inputs_path=f"{_DATASET_STR_FULL}_{_EMBEDDING_VER_STR_FULL}_scoring_inputs.jsonl",
        output_scored_records_path=f"{_DATASET_STR_FULL}_{_MODEL_STR}_{_EMBEDDING_VER_STR_FULL}_scored_records.jsonl",
        output_graph_path=f"{_DATASET_STR_FULL}_{_MODEL_STR}_graph.jsonl",
        errors_output_path=f"{_DATASET_STR_FULL}_{_MODEL_STR}_errors.jsonl",
    )

    def load_helpsteer2():
        return load_dataset("nvidia/HelpSteer2", split="train")

    global_graph = global_pipeline.run(dataset_load_fn=load_helpsteer2, skip_scoring=True)

    print("\n" + "=" * 50)
    print("STEP 2: Building graph for `high-quality` records (Helpfulness = 4)")
    print("=" * 50)

    _EMBEDDING_VER_STR_H4 = "helpfulness_4"
    _DATASET_STR_H4 = "helpsteer2_helpfulness_4"

    helpfulness_4_pipeline = DataValuationPipeline(
        query_formatting_fn=helpsteer2_query_formatter,
        response_formatting_fn=helpsteer2_response_formatter,
        embedding_formatting_fn=helpsteer2_embedding_formatter,
        embedding_model_name=config["embedding_model"],
        prompt_logprobs=config.get("prompt_logprobs", 10),
        vllm_config=config.get("vllm_config"),
        k_neighbors=args.k_neighbors,
        output_embeddings_path=f"{_DATASET_STR_H4}_{_EMB_STR}_{_EMBEDDING_VER_STR_H4}_embeddings.npy",
        output_neighbors_path=f"{_DATASET_STR_H4}_{_EMB_STR}_{_EMBEDDING_VER_STR_H4}_neighbors.npz",
        output_scoring_inputs_path=f"{_DATASET_STR_H4}_{_MODEL_STR}_{_EMBEDDING_VER_STR_H4}_scoring_inputs.jsonl",
        output_scored_records_path=f"{_DATASET_STR_H4}_{_MODEL_STR}_{_EMBEDDING_VER_STR_H4}_scored_records.jsonl",
        output_graph_path=f"{_DATASET_STR_H4}_{_MODEL_STR}_graph.jsonl",
        errors_output_path=f"{_DATASET_STR_H4}_{_MODEL_STR}_errors.jsonl",
        length_normalization=args.length_normalization,
    )

    def load_helpsteer2_helpfulness_4():
        return load_dataset("nvidia/HelpSteer2", split="train").filter(
            lambda x: x["helpfulness"] == 4
        )

    helpfulness_4_graph = helpfulness_4_pipeline.run(
        dataset_load_fn=load_helpsteer2_helpfulness_4,
        skip_scoring=False,
    )

    rule_records = None
    if args.stage1_rule != "adv_delta_threshold":
        rule_records = rank_records_by_rule(
            helpfulness_4_graph,
            args.stage1_rule,
            budget=args.budget,
            min_students=args.min_students,
        )
        anchors_output = args.anchors_output or (
            f"{_DATASET_STR_H4}_{_MODEL_STR}_{args.stage1_rule}_anchors.jsonl"
        )
        with open(anchors_output, "w", encoding="utf-8") as f:
            for rec in rule_records:
                # query_index is the row in the helpfulness-4 subset; also record
                # the row in the full HelpSteer2 train split.
                full_node = global_graph._get_node(rec["query"], rec["response"])
                rec["dataset_row_index"] = full_node.query_index if full_node else None
                f.write(json.dumps({"rule": args.stage1_rule, **rec}) + "\n")
        print(
            f"Wrote {len(rule_records)} Stage-1 records ({args.stage1_rule}) to {anchors_output}"
        )

    if not eval_model:
        print("No --eval_model given; stopping after Step 2.")
        raise SystemExit(0)

    print("\n" + "=" * 50)
    print("STEP 3: Graph Analysis & Contradiction Evaluation")
    print("=" * 50)

    helpsteer2 = load_dataset("nvidia/HelpSteer2", split="train")
    score_lookup = {i: record["helpfulness"] for i, record in enumerate(helpsteer2)}

    if rule_records is not None:
        bad_high_score_records = rule_records
        print(
            f"Flagged {len(bad_high_score_records)} records with the lowest "
            f"{args.stage1_rule} scores."
        )
    else:
        bad_high_score_records = find_high_impact_records(
            helpfulness_4_graph,
            advantage_threshold=args.advantage_threshold,
            min_students=args.min_students,
        )
        print(
            f"Found {len(bad_high_score_records)} records with exemplifying advantage lower than {args.advantage_threshold}"
            + (f" and at least {args.min_students} students." if args.min_students else ".")
        )

    res = find_similar_bad_neighbors(
        graph=global_graph,
        bad_records=bad_high_score_records,
        score_lookup=score_lookup,
        query_embeddings=global_pipeline.output_embeddings_path,
        prompt_similarity_threshold=0,
        response_similarity_threshold=0.5,
        embedding_model_name=config["embedding_model"],
    )
    print(f"Found {len(res)} pairs of similar neighbors.")

    if res:
        prompts = generate_llm_prompts_for_bad_neighbors(
            res,
            data_review_instructions=HELPSTEER_INSTRUCTION,
            include_proposed_response=False,
            # HELPSTEER_INSTRUCTION defines its own JSON schema and refers to
            # [Record A] (high score) / [Record B] (low score) with their scores.
            structured_output=False,
            record_names=("[Record A]", "[Record B]"),
            score_name="Human Helpfulness Score",
        )
        print(
            f"Generated {len(prompts)} prompts to be sent to LLM for review. Sample: {prompts[0]}"
        )

        evaluated = evaluate_contradictions_with_llm(
            prompts,
            api_key=api_key,
            model_name=eval_model,
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

            verdict = model_eval.get("verdict") or model_eval.get("Verdict")
            if verdict == "Annotator A Fault":
                records_with_issue.add(result["record_a_index"])
            elif verdict == "Annotator B Fault":
                records_with_issue.add(result["record_b_index"])

            if model_eval.get("is_contradiction") == True:
                real_contradiction.append(result)

        print(f"Number of real contradiction: {len(real_contradiction)}")
        print(f"Number of records with issue: {len(records_with_issue)}")
        print("\n=== Evaluation Results ===")

        with open(args.report_output, "w", encoding="utf-8") as f_out:
            for idx, result in enumerate(real_contradiction):
                model_eval = result.get("llm_evaluation", {})
                if isinstance(model_eval, list) and len(model_eval) == 1:
                    model_eval = model_eval[0]
                if model_eval.get("is_contradiction") == True:
                    print(
                        f"\n\033[1m-------- Contradiction Pair {idx + 1} --------\033[0m"
                    )
                    print(
                        f"\033[1mRecord 1 (Index {result['record_a_index']}):\033[0m \n"
                    )
                    print(
                        f"\033[1mQuery:\033[0m {result['record_a_query']} \033[1m \nResponse:\033[0m {result['record_a_response']}"
                    )
                    print(
                        f"\033[1mHelpfulness Score:\033[0m {score_lookup.get(result['record_a_index'])}"
                    )
                    print(
                        f"\033[1mRecord 2 (Index {result['record_b_index']}):\033[0m \n"
                    )
                    print(
                        f"\033[1mQuery:\033[0m {result['record_b_query']} \033[1m \nResponse:\033[0m {result['record_b_response']}"
                    )
                    print(
                        f"\033[1mHelpfulness Score:\033[0m {score_lookup.get(result['record_b_index'])}"
                    )
                    print("\n\033[1mLLM Verdict:\033[0m")
                    print(
                        f"\033[1mIs Contradiction:\033[0m {model_eval.get('is_contradiction')}"
                    )
                    print(f"\033[1mVerdict:\033[0m {model_eval.get('verdict')}")
                    print(f"\033[1mReasoning:\033[0m {model_eval.get('reasoning')}")

                    f_out.write(f"\n-------- Contradiction Pair {idx + 1} --------\n")
                    f_out.write(f"Record 1 (Index {result['record_a_index']}):\n")
                    f_out.write(
                        f"Query: {result['record_a_query']} \nResponse: {result['record_a_response']}\n"
                    )
                    f_out.write(
                        f"Helpfulness Score: {score_lookup.get(result['record_a_index'])}\n\n"
                    )
                    f_out.write(f"Record 2 (Index {result['record_b_index']}):\n")
                    f_out.write(
                        f"Query: {result['record_b_query']} \nResponse: {result['record_b_response']}\n"
                    )
                    f_out.write(
                        f"Helpfulness Score: {score_lookup.get(result['record_b_index'])}\n\n"
                    )
                    f_out.write(f"LLM Verdict:\n")
                    f_out.write(
                        f"Is Contradiction: {model_eval.get('is_contradiction')}\n"
                    )
                    f_out.write(f"Verdict: {model_eval.get('verdict')}\n")
                    f_out.write(f"Reasoning: {model_eval.get('reasoning')}\n")
        print(f"\nSaved formatted contradiction report to {args.report_output}")
    else:
        print("No similar bad neighbors found, skipping evaluation.")
