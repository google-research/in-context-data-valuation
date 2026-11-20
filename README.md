# In-Context Data Valuation

Training-free data valuation and curation for LLM text datasets, accompanying
the paper
[First-Order In-Context Shapley: Scalable Data Auditing for LLM Alignment and Evaluation](https://arxiv.org/abs/2607.22766).

The pipeline builds a k-NN influence graph in which each edge measures how much
a neighbor, shown as a one-shot demonstration, changes the log-likelihood of a
record's response. From this graph it computes per-record metrics (Exemplified
Advantage `Adv_e`, Exemplifying Advantage `Adv_delta`, One-of-a-Kind score),
flags candidate label errors, and optionally has an LLM arbitrate each flagged
record against a contrasting neighbor. See the paper for the definitions and
the two-stage audit procedure.

This is not an officially supported Google product. Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

This project is intended for demonstration purposes only. It is not intended for
use in a production environment.

## Repository layout

| File | Purpose |
| :--- | :--- |
| `valuation_pipeline.py` | `DataValuationPipeline`: load, embed, find neighbors, score with vLLM, build the graph |
| `neighbor_finding.py` | Embeddings (`sentence-transformers`) and HNSW nearest neighbors (`hnswlib`) |
| `scoring.py` | 0-shot / 1-shot prompts and batched log-likelihoods |
| `influence_graph.py` | `QueryInfluenceGraph` and the per-record metrics |
| `curation_rules.py` | Metric-based curation rules and `curate_graph` |
| `graph_analysis.py` | Stage-1 candidate selection, arbitration prompts, LLM arbitration via `litellm` |
| `e2e_helpsteer2_valuation.py`, `e2e_hh_rlhf_valuation.py` | End-to-end examples |
| `default.json` | Default embedding model, scoring model and vLLM settings |

## Setup

```bash
docker build -t in-context-data-valuation .
```

The image is based on `vllm/vllm-openai:v0.29.0` and installs the pinned
`requirements.txt`. Notes:

* The pinned vLLM/PyTorch build targets CUDA 13, so the host needs a
  CUDA 13–compatible NVIDIA driver.
* Outside Docker, use Python 3.11 or newer.
* `default.json` uses `tensor_parallel_size: 4`; adjust `vllm_config` for your
  GPUs.
* Arbitration works with any [litellm provider](https://github.com/BerriAI/litellm#supported-providers-website-supported-models--docs).
  Pass `--api_key` or set `LLM_API_KEY`. Verdicts are cached in `--eval_output`,
  so reruns only call the API for new pairs.

## Quick start

HelpSteer2 (graded labels):

```bash
docker run --gpus all --shm-size=2g -e LLM_API_KEY=$LLM_API_KEY -v "$(pwd):/app" \
  in-context-data-valuation:latest \
  python e2e_helpsteer2_valuation.py --config default.json --eval_model "<litellm model>"
```

HH-RLHF (implicit labels), training or test split:

```bash
docker run --gpus all --shm-size=2g -e LLM_API_KEY=$LLM_API_KEY -v "$(pwd):/app" \
  in-context-data-valuation:latest \
  python e2e_hh_rlhf_valuation.py --config default.json --eval_model "<litellm model>"

python e2e_hh_rlhf_valuation.py --config default.json --data_split test --budget 450 \
  --eval_output hh_rlhf_test_evaluations.jsonl \
  --report_output hh_rlhf_test_contradictions_report.txt --eval_model "<litellm model>"
```

Without `--eval_model`, the scripts stop after saving the influence graph. Each
script writes its arbitration verdicts (`--eval_output`) and a readable report
of confirmed contradictions (`--report_output`). Run a script with `--help` for
all options; the most useful ones:

| Option | Meaning |
| :--- | :--- |
| `--stage1_rule` | How Stage 1 picks candidates: the paper's rule by default (`adv_e_deviation` for HH-RLHF, `adv_delta_threshold` for HelpSteer2), or a curation rule (`calibrated_margin`, `exemplifying_advantage`) applied to the `--budget` lowest-scoring records. With a rule, the selected records are also written to `--anchors_output`. |
| `--budget` | Number of Stage-1 candidates. |
| `--score_both_labels` | HH-RLHF only: score both option labels and use the margin `log p(observed) - log p(other)` as each score (doubles scoring cost). |
| `--pairing_mode` | HH-RLHF only: `worst_neighbor` (default) or `mutual_nn`. |
| `--sample_size N` | HH-RLHF only: use the first N records, for a quick test run. |
| `--k_neighbors` | Neighborhood size including the record itself (default 16). |

## Curating a saved graph

```python
from curation_rules import curate_graph
from influence_graph import QueryInfluenceGraph

graph = QueryInfluenceGraph.load_from_jsonl("hh_rlhf_train_qwen3.5-9b_full_graph.jsonl")
result = curate_graph(graph, rule="calibrated_margin", target_percentage=0.9)
kept_row_indices = sorted(node.query_index for node in result.kept_nodes)
```

Registered rules: `exemplified_advantage_outlier` and `exemplifying_advantage`
(the paper's metrics), and `calibrated_margin`. See the docstrings in
`curation_rules.py` for options such as the uniqueness guard and the
`calibrated_margin` weight.

## Running unit tests

The unit tests run on CPU without `vllm` or a GPU:

```bash
python3 -m unittest discover -s tests -v
```

## Contributors

* Jun Qin
* Matthew Watson
* Moustafa Farid Alzantot
* Volodya Shtenovych
* Yunting Song
