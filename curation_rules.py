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

"""Metric-based dataset curation rules, including response-calibrated margin.

Provides pluggable `FilteringRule` implementations that operate directly on a
`QueryInfluenceGraph` to flag metric outliers and rank records for subset
selection:

  * `CalibratedMarginRule` (`"calibrated_margin"`): response-stratified
    zero-shot margin plus neighborhood ratio Z-score with 1-sided lower-tail
    outlier pruning and a positive-influence one-of-a-kind guard.
  * `ExemplifiedAdvantageOutlierRule` (`"exemplified_advantage_outlier"`):
    2-sided `|Adv_e - median(Adv_e)| > k * std(Adv_e)` outlier pruning followed
    by `Adv_delta` ranking.
  * `ExemplifyingAdvantageRule` (`"exemplifying_advantage"`): direct ranking by
    `exemplifying_advantage`.
"""

import abc
from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
import math
import statistics
import types
from typing import Any, NamedTuple

from influence_graph import QueryInfluenceGraph, _QueryNode

NO_OP_FILTERING_RULE = "none"
DEFAULT_FILTERING_RULE = NO_OP_FILTERING_RULE

_METRIC_SCORE_FIELDS = frozenset(
    (
        "base_score",
        "exemplified_score_ratio",
        "exemplifying_score_ratio",
        "one_of_a_kind_ratio",
        "exemplified_advantage",
        "exemplified_mean",
        "exemplified_std",
        "exemplifying_advantage",
        "exemplifying_mean",
        "exemplifying_std",
    )
)

_MAX_STRATUM_KEY_CHARS = 512
_MAX_TRACKED_STRATA = 1024
_MAX_ACTIVE_STRATA = 256
_MIN_STRATUM_SIZE = 2
_MIN_STRATIFIED_COVERAGE = 0.5
_GLOBAL_STRATUM = "__global__"


class MetricSummary(NamedTuple):
    """Summary statistics for a single valuation metric.

    Attributes:
      count: Number of nodes with a finite value for this metric.
      mean: Arithmetic mean; NaN when `count` is zero.
      median: Exact median; NaN when `count` is zero.
      std: Population standard deviation; NaN when `count` is zero.
    """

    count: int = 0
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")


CorpusStats = Mapping[str, MetricSummary]


class StratifiedCorpusStats(NamedTuple):
    """Corpus-wide and optional per-response-stratum metric summary statistics.

    Attributes:
      global_stats: Corpus-wide `CorpusStats` across all scored graph nodes.
      by_stratum: Per-response `CorpusStats` for active discrete response
        strata. Empty when stratification is disabled or fell back to global.
      active: True when discrete response stratification is active.
    """

    global_stats: CorpusStats
    by_stratum: Mapping[str, CorpusStats] = types.MappingProxyType({})
    active: bool = False

    def stats_for(self, stratum: str) -> CorpusStats:
        """Returns stratum stats when active and present, else `global_stats`."""
        if self.active and stratum in self.by_stratum:
            return self.by_stratum[stratum]
        return self.global_stats


class UniquenessGuard(str, enum.Enum):
    """Criterion for protecting rare records from metric-based curation.

    Records meeting the criterion ("unique") are spared from `FilteringRule`
    exclusion (but not from LLM contradiction pruning) and always kept during
    subset selection (see `curate_graph`). Each rule has a default guard; pass
    another one to the rule's constructor, e.g.
    `CalibratedMarginRule(uniqueness_guard=UniquenessGuard.NONE)`.
    """

    ONE_OF_A_KIND = "one_of_a_kind"
    ONE_OF_A_KIND_POSITIVE_INFLUENCE = "one_of_a_kind_positive_influence"
    NONE = "none"


@dataclasses.dataclass(frozen=True)
class FilteringRule(abc.ABC):
    """Abstract base class for pluggable metric-based curation rules."""

    required_metrics: tuple[str, ...] = ()
    uniqueness_guard: UniquenessGuard = UniquenessGuard.ONE_OF_A_KIND
    stratify_by_response: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.uniqueness_guard, UniquenessGuard):
            object.__setattr__(
                self, "uniqueness_guard", UniquenessGuard(self.uniqueness_guard)
            )

    def is_unique(
        self,
        node: _QueryNode,
        one_of_a_kind_threshold: float,
    ) -> bool:
        """Returns True if `node` is protected by `uniqueness_guard`."""
        ratio = node.one_of_a_kind_ratio
        is_one_of_a_kind = (
            ratio is not None
            and math.isfinite(ratio)
            and ratio >= one_of_a_kind_threshold
        )
        adv = node.exemplifying_advantage
        if self.uniqueness_guard == UniquenessGuard.ONE_OF_A_KIND:
            return is_one_of_a_kind
        if self.uniqueness_guard == UniquenessGuard.ONE_OF_A_KIND_POSITIVE_INFLUENCE:
            return (
                is_one_of_a_kind
                and adv is not None
                and math.isfinite(adv)
                and adv > 0.0
            )
        if self.uniqueness_guard == UniquenessGuard.NONE:
            return False
        raise ValueError(f"Unknown uniqueness_guard {self.uniqueness_guard!r}.")

    @abc.abstractmethod
    def is_excluded(self, node: _QueryNode, stats: CorpusStats) -> bool:
        """Returns True if a non-unique `node` should be excluded as an outlier."""

    @abc.abstractmethod
    def rank_score(self, node: _QueryNode, stats: CorpusStats) -> float | None:
        """Returns a scalar ranking score (higher is better), or None."""


_FILTERING_RULES: dict[str, FilteringRule] = {}


def register_filtering_rule(
    name: str,
    **kwargs: Any,
) -> Callable[[type[FilteringRule]], type[FilteringRule]]:
    """Class decorator that registers a `FilteringRule` under `name`."""
    if not name:
        raise ValueError("Filtering rule name must be non-empty.")

    def _decorator(cls: type[FilteringRule]) -> type[FilteringRule]:
        if name in _FILTERING_RULES:
            raise ValueError(f"Filtering rule {name!r} is already registered.")
        rule = cls(**kwargs)
        for metric in rule.required_metrics:
            if metric not in _METRIC_SCORE_FIELDS:
                raise ValueError(
                    f"Rule {name!r} declares unknown metric field {metric!r}; "
                    f"must be one of {sorted(_METRIC_SCORE_FIELDS)}."
                )
        _FILTERING_RULES[name] = rule
        return cls

    return _decorator


def get_filtering_rule(name: str) -> FilteringRule:
    """Returns the registered `FilteringRule` for `name`."""
    rule = _FILTERING_RULES.get(name)
    if rule is None:
        available = ", ".join(registered_filtering_rules())
        raise ValueError(
            f"Unknown filtering_rule {name!r}. Available rules: [{available}]."
        )
    return rule


def registered_filtering_rules() -> tuple[str, ...]:
    """Returns the sorted names of all registered `FilteringRule`s."""
    return tuple(sorted(_FILTERING_RULES))


def get_metric_stats(stats: CorpusStats, metric_name: str) -> MetricSummary:
    """Returns the non-empty `MetricSummary` for `metric_name` from `stats`."""
    if metric_name not in _METRIC_SCORE_FIELDS:
        raise ValueError(
            f"Unknown metric field {metric_name!r}; must be one of "
            f"{sorted(_METRIC_SCORE_FIELDS)}."
        )
    summary = stats.get(metric_name)
    if summary is None or summary.count <= 0:
        raise ValueError(
            f"Corpus stats for {metric_name!r} are missing or empty "
            f"({summary!r}). Ensure the metric is listed in `required_metrics` "
            "and has finite values in the influence graph."
        )
    return summary


@register_filtering_rule(NO_OP_FILTERING_RULE)
class NoOpRule(FilteringRule):
    """No-op metric rule that neither excludes nor ranks records."""

    def is_excluded(self, node: _QueryNode, stats: CorpusStats) -> bool:
        del node, stats
        return False

    def rank_score(self, node: _QueryNode, stats: CorpusStats) -> float | None:
        del node, stats
        return None


@register_filtering_rule("exemplifying_advantage")
@dataclasses.dataclass(frozen=True)
class ExemplifyingAdvantageRule(FilteringRule):
    """Ranks by `exemplifying_advantage` without pruning."""

    def is_excluded(self, node: _QueryNode, stats: CorpusStats) -> bool:
        del node, stats
        return False

    def rank_score(self, node: _QueryNode, stats: CorpusStats) -> float | None:
        del stats
        return node.exemplifying_advantage


@register_filtering_rule("exemplified_advantage_outlier")
@dataclasses.dataclass(frozen=True)
class ExemplifiedAdvantageOutlierRule(ExemplifyingAdvantageRule):
    """Prunes `|Adv_e - median(Adv_e)| > k * std(Adv_e)` before ranking."""

    required_metrics: tuple[str, ...] = ("exemplified_advantage",)
    outlier_sigmas: float = 2.0

    def is_excluded(self, node: _QueryNode, stats: CorpusStats) -> bool:
        adv_stats = get_metric_stats(stats, "exemplified_advantage")
        if not (adv_stats.std > 0):
            return False
        adv = node.exemplified_advantage
        if adv is None or not math.isfinite(adv):
            return False
        return abs(adv - adv_stats.median) > self.outlier_sigmas * adv_stats.std


@register_filtering_rule("calibrated_margin")
@dataclasses.dataclass(frozen=True)
class CalibratedMarginRule(FilteringRule):
    """Response-calibrated zero-shot margin plus neighborhood ratio Z-score.

    Zero-shot `base_score` values carry response-prior and length biases across
    discrete labels, so uncalibrated global ranking can starve lower-prior
    classes. Both `base_score` (`B = s_0(i)`) and `exemplified_score_ratio`
    (`R = mean_s_1(i) - s_0(i)`) share additive scoring units (for example,
    nats), so `B + w * R = (1 - w) * s_0(i) + w * mean_s_1(i)`.

    When paired with `stratify_by_response=True`, this rule centers `B` and `R`
    against each response stratum's mean and normalizes by
    `hypot(std_base, w * std_ratio)`:

    * **1-sided lower-tail exclusion**: Only strongly negative calibrated
      Z-scores (`z < -outlier_sigmas`) are excluded; upper-tail records are
      unambiguous prototypes and are retained first.
    * **Stratum-mean missing-ratio imputation**: Isolated queries with no
      incoming neighbors (`exemplified_score_ratio` missing or non-finite)
      impute the stratum mean (`ratio - mean_ratio = 0.0`).
    * **Positive-influence uniqueness guard by default**: Prevents negative-
      influence anti-hubs (`exemplifying_advantage <= 0` or missing) from
      bypassing 1-sided pruning or displacing ranked records.
    """

    required_metrics: tuple[str, ...] = (
        "base_score",
        "exemplified_score_ratio",
    )
    uniqueness_guard: UniquenessGuard = (
        UniquenessGuard.ONE_OF_A_KIND_POSITIVE_INFLUENCE
    )
    stratify_by_response: bool = True
    outlier_sigmas: float = 2.0
    exemplified_ratio_weight: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.outlier_sigmas) or self.outlier_sigmas <= 0:
            raise ValueError(
                "`outlier_sigmas` must be finite and positive, got "
                f"{self.outlier_sigmas}."
            )
        if (
            not math.isfinite(self.exemplified_ratio_weight)
            or self.exemplified_ratio_weight < 0
        ):
            raise ValueError(
                "`exemplified_ratio_weight` must be finite and non-negative, "
                f"got {self.exemplified_ratio_weight}."
            )

    def _calibrated_margin_and_scale(
        self,
        node: _QueryNode,
        stats: CorpusStats,
    ) -> tuple[float, float] | None:
        """Computes `(calibrated_margin, combined_std_scale)` for `node`."""
        if node.base_score is None or not math.isfinite(node.base_score):
            return None
        base_stats = get_metric_stats(stats, "base_score")
        if not math.isfinite(base_stats.mean):
            return None
        calibrated = node.base_score - base_stats.mean

        ratio_stats = stats.get("exemplified_score_ratio")
        ratio = node.exemplified_score_ratio
        if (
            ratio is not None
            and math.isfinite(ratio)
            and ratio_stats is not None
            and ratio_stats.count > 0
            and math.isfinite(ratio_stats.mean)
        ):
            calibrated += self.exemplified_ratio_weight * (
                ratio - ratio_stats.mean
            )

        b_std = (
            base_stats.std
            if base_stats.std > 0 and math.isfinite(base_stats.std)
            else 0.0
        )
        if (
            ratio_stats is not None
            and ratio_stats.count > 0
            and math.isfinite(ratio_stats.mean)
            and ratio_stats.std > 0
            and math.isfinite(ratio_stats.std)
        ):
            r_std = self.exemplified_ratio_weight * ratio_stats.std
        else:
            r_std = 0.0
        scale = math.hypot(b_std, r_std)
        return (calibrated, scale)

    def is_excluded(self, node: _QueryNode, stats: CorpusStats) -> bool:
        margin_and_scale = self._calibrated_margin_and_scale(node, stats)
        if margin_and_scale is None:
            return False
        calibrated, scale = margin_and_scale
        if not (scale > 0 and math.isfinite(scale)):
            return False
        return (calibrated / scale) < -self.outlier_sigmas

    def rank_score(self, node: _QueryNode, stats: CorpusStats) -> float | None:
        margin_and_scale = self._calibrated_margin_and_scale(node, stats)
        if margin_and_scale is None:
            return None
        calibrated, scale = margin_and_scale
        if not (scale > 0 and math.isfinite(scale)):
            return 0.0
        return calibrated / scale


def _summarize_values(values: Sequence[float]) -> MetricSummary:
    """Computes `MetricSummary` (population std) over finite `values`."""
    if not values:
        return MetricSummary()
    n = len(values)
    mean = sum(values) / n
    median = float(statistics.median(values))
    var = sum((v - mean) ** 2 for v in values) / n
    return MetricSummary(count=n, mean=mean, median=median, std=math.sqrt(var))


def _compute_stats_for_nodes(
    nodes: Sequence[_QueryNode],
    metrics: Sequence[str],
) -> dict[str, MetricSummary]:
    """Computes `CorpusStats` for each requested metric over `nodes`."""
    stats: dict[str, MetricSummary] = {}
    for metric in metrics:
        vals = []
        for node in nodes:
            v = getattr(node, metric, None)
            if v is not None and math.isfinite(v):
                vals.append(float(v))
        stats[metric] = _summarize_values(vals)
    return stats


def _response_stratum_key(response: str) -> str | None:
    """Returns canonical response stratum key, or None if ineligible."""
    if len(response) > _MAX_STRATUM_KEY_CHARS:
        return None
    return response


def compute_stratified_corpus_stats(
    nodes: Sequence[_QueryNode],
    required_metrics: Sequence[str],
    *,
    stratify_by_response: bool = False,
) -> StratifiedCorpusStats:
    """Computes corpus-wide and (when eligible) per-response-stratum stats.

    Response stratification activates when `stratify_by_response=True`, there are
    between 2 and `_MAX_ACTIVE_STRATA` (256) recurring response strata of size
    `>= 2` (with response length `<= 512` chars), and those strata cover at
    least 50% of all nodes. Otherwise falls back to unstratified global stats.
    """
    global_stats = _compute_stats_for_nodes(nodes, required_metrics)
    if not stratify_by_response or not nodes:
        return StratifiedCorpusStats(
            global_stats=global_stats, by_stratum={}, active=False
        )

    grouped: dict[str, list[_QueryNode]] = {}
    for node in nodes:
        key = _response_stratum_key(node.response)
        if key is None:
            continue
        if key not in grouped and len(grouped) >= _MAX_TRACKED_STRATA:
            return StratifiedCorpusStats(
                global_stats=global_stats, by_stratum={}, active=False
            )
        grouped.setdefault(key, []).append(node)

    valid_strata = [
        stratum
        for stratum, stratum_nodes in grouped.items()
        if len(stratum_nodes) >= _MIN_STRATUM_SIZE
    ]
    covered = sum(len(grouped[s]) for s in valid_strata)
    active = (
        2 <= len(valid_strata) <= _MAX_ACTIVE_STRATA
        and covered >= _MIN_STRATIFIED_COVERAGE * len(nodes)
    )
    if not active:
        return StratifiedCorpusStats(
            global_stats=global_stats, by_stratum={}, active=False
        )

    by_stratum: dict[str, CorpusStats] = {}
    for stratum in valid_strata:
        raw_stats = _compute_stats_for_nodes(grouped[stratum], required_metrics)
        by_stratum[stratum] = {
            **global_stats,
            **{
                metric: summary
                for metric, summary in raw_stats.items()
                if summary.count >= _MIN_STRATUM_SIZE
            },
        }
    return StratifiedCorpusStats(
        global_stats=global_stats, by_stratum=by_stratum, active=True
    )


@dataclasses.dataclass
class CurationResult:
    """Output of applying a `FilteringRule` to a `QueryInfluenceGraph`.

    Attributes:
      kept_nodes: Nodes retained in the curated subset.
      excluded_nodes: Nodes dropped by LLM exclusion, rule exclusion, or top-K
        cutoff.
      llm_excluded_keys: `(query, response)` keys dropped by `llm_excluded_keys`.
      rule_excluded_keys: `(query, response)` keys flagged by `rule.is_excluded`
        (excluding unique nodes spared by `uniqueness_guard`).
      unique_kept_keys: `(query, response)` keys preserved via
        `uniqueness_guard`.
      rank_scores: Mapping from `(query, response)` to its computed rule
        `rank_score` (or None).
      stats: Computed `StratifiedCorpusStats`.
    """

    kept_nodes: list[_QueryNode]
    excluded_nodes: list[_QueryNode]
    llm_excluded_keys: set[tuple[str, str]]
    rule_excluded_keys: set[tuple[str, str]]
    unique_kept_keys: set[tuple[str, str]]
    rank_scores: dict[tuple[str, str], float | None]
    stats: StratifiedCorpusStats


def curate_graph(
    graph: QueryInfluenceGraph,
    rule: str | FilteringRule = "calibrated_margin",
    *,
    target_percentage: float = 1.0,
    one_of_a_kind_threshold: float = 0.9,
    llm_excluded_keys: Iterable[tuple[str, str]] | None = None,
) -> CurationResult:
    """Applies a `FilteringRule` to all nodes in `graph`.

    Filtering order:
      1. Records in `llm_excluded_keys` are excluded unconditionally.
      2. Remaining records passing `rule.is_unique(node, one_of_a_kind_threshold)`
         are spared from rule exclusion and always kept. They count toward
         `target_percentage`, but are kept even if they exceed it, so the kept
         fraction can be above `target_percentage` when many records are unique.
      3. Non-unique records failing `rule.is_excluded` are pruned; surviving
         candidates are ranked by descending `rule.rank_score` (tie-breaking by
         `query_index`) to fill the rest of `target_percentage`.

    Args:
      graph: A populated `QueryInfluenceGraph`.
      rule: Registered rule name (e.g. `"calibrated_margin"`) or `FilteringRule`
        instance.
      target_percentage: Fraction in `(0, 1]` of all graph nodes to retain
        (see step 2 for unique records).
      one_of_a_kind_threshold: Threshold for `one_of_a_kind_ratio`.
      llm_excluded_keys: Optional iterable of `(query, response)` keys already
        confirmed bad by LLM contradiction arbitration.

    Returns:
      A `CurationResult` containing kept/excluded nodes and diagnostics.
    """
    if not (0.0 < target_percentage <= 1.0):
        raise ValueError(
            f"`target_percentage` must be in (0, 1], got {target_percentage}."
        )
    active_rule = (
        get_filtering_rule(rule) if isinstance(rule, str) else rule
    )
    if (
        "base_score" in active_rule.required_metrics
        and "exemplified_score_ratio" in active_rule.required_metrics
        and graph.length_normalization != "none"
    ):
        raise ValueError(
            f"Rule {type(active_rule).__name__} adds `base_score` (total nats) to "
            "`exemplified_score_ratio`, which is length-normalized when the graph's "
            f"length_normalization is {graph.length_normalization!r}; the units "
            "would not match. Build the graph with length_normalization='none'."
        )
    graph._recalculate_dirty_nodes()

    all_nodes: list[_QueryNode] = [
        node
        for query_dict in graph.nodes.values()
        for node in query_dict.values()
    ]
    stats = compute_stratified_corpus_stats(
        all_nodes,
        active_rule.required_metrics,
        stratify_by_response=active_rule.stratify_by_response,
    )

    llm_excluded = set(llm_excluded_keys or ())
    applied_llm_excluded: set[tuple[str, str]] = set()
    rule_excluded: set[tuple[str, str]] = set()
    unique_kept: set[tuple[str, str]] = set()
    rank_scores: dict[tuple[str, str], float | None] = {}

    def _stratum_of(node: _QueryNode) -> str:
        if stats.active and node.response in stats.by_stratum:
            return node.response
        return _GLOBAL_STRATUM

    uniques: list[_QueryNode] = []
    candidates: list[_QueryNode] = []
    excluded_nodes: list[_QueryNode] = []

    for node in all_nodes:
        key = (node.query, node.response)
        node_stats = stats.stats_for(_stratum_of(node))
        rank_scores[key] = active_rule.rank_score(node, node_stats)

        if key in llm_excluded:
            applied_llm_excluded.add(key)
            excluded_nodes.append(node)
            continue

        if active_rule.is_unique(node, one_of_a_kind_threshold):
            unique_kept.add(key)
            uniques.append(node)
            continue

        if active_rule.is_excluded(node, node_stats):
            rule_excluded.add(key)
            excluded_nodes.append(node)
            continue

        candidates.append(node)

    def _sort_key(n: _QueryNode) -> tuple[int, float, int]:
        s = rank_scores[(n.query, n.response)]
        if s is None or not math.isfinite(s):
            return (1, 0.0, n.query_index)
        return (0, -s, n.query_index)

    target = round(len(all_nodes) * target_percentage)
    needed = max(0, target - len(uniques))
    candidates.sort(key=_sort_key)
    kept_nodes: list[_QueryNode] = uniques + candidates[:needed]
    excluded_nodes.extend(candidates[needed:])

    return CurationResult(
        kept_nodes=kept_nodes,
        excluded_nodes=excluded_nodes,
        llm_excluded_keys=applied_llm_excluded,
        rule_excluded_keys=rule_excluded,
        unique_kept_keys=unique_kept,
        rank_scores=rank_scores,
        stats=stats,
    )
