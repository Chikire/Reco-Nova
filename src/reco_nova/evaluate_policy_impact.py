"""Estimate business-impact lift by simulating interactions for two policies.

This module compares a baseline policy (popularity or random) with a
personalized hybrid policy using holdout relevance labels and a simple
position-biased click simulator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reco_nova.evaluation import OfflineMetrics, evaluate_rankings
from reco_nova.models import (
    CollaborativeSVD,
    ContentRecommender,
    HybridRecommender,
    PopularityRecommender,
)
from reco_nova.train import (
    _positive_limit,
    build_ground_truth,
    read_interactions,
)


@dataclass(frozen=True)
class SimulatedOutcomes:
    """Aggregate outcomes from simulated recommendation sessions."""

    impressions: int
    clicks: int
    relevant_clicks: int
    sessions: int
    sessions_with_click: int
    sessions_with_relevant_click: int

    def to_dict(self) -> dict[str, int | float]:
        impressions = max(self.impressions, 1)
        sessions = max(self.sessions, 1)
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "relevant_clicks": self.relevant_clicks,
            "sessions": self.sessions,
            "click_through_rate": self.clicks / impressions,
            "relevant_click_through_rate": self.relevant_clicks / impressions,
            "avg_clicks_per_session": self.clicks / sessions,
            "sessions_with_click_rate": self.sessions_with_click / sessions,
            "sessions_with_relevant_click_rate": (
                self.sessions_with_relevant_click / sessions
            ),
        }


class RandomPolicy:
    """Recommend unseen items uniformly at random for each user."""

    def __init__(
        self,
        item_ids: list[str],
        seen_by_user: dict[str, set[str]],
        rng: np.random.Generator,
    ) -> None:
        self.item_ids = np.array(item_ids, dtype=object)
        self.seen_by_user = seen_by_user
        self.rng = rng

    def recommend(self, user_id: str, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        seen = self.seen_by_user.get(str(user_id), set())
        eligible = [item for item in self.item_ids if item not in seen]
        if not eligible:
            return []
        self.rng.shuffle(eligible)
        picked = eligible[:k]
        # Constant score keeps interface aligned with other policies.
        return [(str(item), 1.0) for item in picked]


def _recommend_users(
    model: object,
    users: list[str],
    k: int,
) -> dict[str, list[str]]:
    return {
        user: [
            item for item, _ in model.recommend(user, k)
        ]  # type: ignore[attr-defined]
        for user in users
    }


def _metric_key(metrics: OfflineMetrics) -> tuple[float, float, float]:
    return (metrics.ndcg_at_k, metrics.map_at_k, metrics.hit_rate_at_k)


def _tune_hybrid_weight(
    collaborative: CollaborativeSVD,
    content: ContentRecommender,
    truth: dict[str, set[str]],
    users: list[str],
    k: int,
    hybrid_weights: tuple[float, ...],
) -> tuple[float, dict[str, dict[str, int | float]]]:
    best_weight = hybrid_weights[0]
    best_metrics: OfflineMetrics | None = None
    tuning: dict[str, dict[str, int | float]] = {}
    for weight in hybrid_weights:
        model = HybridRecommender(collaborative, content, weight)
        recommendations = _recommend_users(model, users, k)
        metrics = evaluate_rankings(recommendations, truth, k)
        tuning[f"cf_{weight:.2f}"] = metrics.to_dict()
        if (
            best_metrics is None
            or _metric_key(metrics) > _metric_key(best_metrics)
        ):
            best_metrics = metrics
            best_weight = weight
    return best_weight, tuning


def _simulate_clicks(
    recommendations: dict[str, list[str]],
    truth: dict[str, set[str]],
    rounds: int,
    rng: np.random.Generator,
    relevant_click_base: float,
    irrelevant_click_base: float,
) -> SimulatedOutcomes:
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")

    users = sorted(set(recommendations) & set(truth))
    impressions = 0
    clicks = 0
    relevant_clicks = 0
    sessions_with_click = 0
    sessions_with_relevant_click = 0

    for user in users:
        ranked = recommendations[user]
        relevant = truth[user]
        for _ in range(rounds):
            session_clicked = False
            session_relevant = False
            for rank, item in enumerate(ranked, start=1):
                # Position bias: higher ranks are more likely to be examined.
                position_weight = 1.0 / np.log2(rank + 1)
                base = (
                    relevant_click_base
                    if item in relevant
                    else irrelevant_click_base
                )
                prob = min(max(base * position_weight, 0.0), 1.0)
                impressions += 1
                if rng.random() < prob:
                    clicks += 1
                    session_clicked = True
                    if item in relevant:
                        relevant_clicks += 1
                        session_relevant = True
            if session_clicked:
                sessions_with_click += 1
            if session_relevant:
                sessions_with_relevant_click += 1

    return SimulatedOutcomes(
        impressions=impressions,
        clicks=clicks,
        relevant_clicks=relevant_clicks,
        sessions=len(users) * rounds,
        sessions_with_click=sessions_with_click,
        sessions_with_relevant_click=sessions_with_relevant_click,
    )


def _lift(personalized: float, baseline: float) -> float:
    if baseline == 0:
        return float("inf") if personalized > 0 else 0.0
    return (personalized - baseline) / baseline


def _build_markdown(report: dict[str, object]) -> str:
    config = report["configuration"]
    data = report["data"]
    baseline_metrics = report["offline_metrics"]["baseline"]
    personalized_metrics = report["offline_metrics"]["personalized_hybrid"]
    baseline_sim = report["simulated_outcomes"]["baseline"]
    personalized_sim = report["simulated_outcomes"]["personalized_hybrid"]
    lift = report["lift"]

    lines = [
        "# Reco-Nova Policy Impact Simulation",
        "",
        (
            "This report estimates business impact by comparing a baseline "
            "policy "
        ),
        (
            "against a personalized hybrid policy under simulated user "
            "interactions."
        ),
        "",
        "## Setup",
        "",
        f"- Baseline policy: {config['baseline_policy']}",
        "- Personalized policy: tuned hybrid recommender",
        f"- Training rows used: {data['training_rows']:,}",
        f"- Warm test users simulated: {data['warm_test_users']:,}",
        f"- Ranking cutoff: K={config['k']}",
        f"- Simulation rounds per user: {config['simulation_rounds']}",
        f"- Relevant click base probability: {config['relevant_click_base']}",
        (
            "- Irrelevant click base probability: "
            f"{config['irrelevant_click_base']}"
        ),
        "",
        "## Offline Ranking Metrics",
        "",
        "| Policy | NDCG@K | MAP@K | Hit Rate@K |",
        "|---|---:|---:|---:|",
        (
            f"| baseline ({config['baseline_policy']}) | "
            f"{baseline_metrics['ndcg_at_k']:.6f} | "
            f"{baseline_metrics['map_at_k']:.6f} | "
            f"{baseline_metrics['hit_rate_at_k']:.6f} |"
        ),
        (
            "| personalized_hybrid | "
            f"{personalized_metrics['ndcg_at_k']:.6f} | "
            f"{personalized_metrics['map_at_k']:.6f} | "
            f"{personalized_metrics['hit_rate_at_k']:.6f} |"
        ),
        "",
        "## Simulated Interaction Outcomes",
        "",
        (
            "| Policy | CTR | Relevant CTR | Sessions w/ Click | Sessions w/ "
            "Relevant Click | Avg Clicks/Session |"
        ),
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| baseline ({config['baseline_policy']}) | "
            f"{baseline_sim['click_through_rate']:.6f} | "
            f"{baseline_sim['relevant_click_through_rate']:.6f} | "
            f"{baseline_sim['sessions_with_click_rate']:.6f} | "
            f"{baseline_sim['sessions_with_relevant_click_rate']:.6f} | "
            f"{baseline_sim['avg_clicks_per_session']:.6f} |"
        ),
        (
            "| personalized_hybrid | "
            f"{personalized_sim['click_through_rate']:.6f} | "
            f"{personalized_sim['relevant_click_through_rate']:.6f} | "
            f"{personalized_sim['sessions_with_click_rate']:.6f} | "
            f"{personalized_sim['sessions_with_relevant_click_rate']:.6f} | "
            f"{personalized_sim['avg_clicks_per_session']:.6f} |"
        ),
        "",
        "## Lift (Personalized vs Baseline)",
        "",
        f"- CTR lift: {lift['click_through_rate_lift']:.2%}",
        (
            "- Relevant CTR lift: "
            f"{lift['relevant_click_through_rate_lift']:.2%}"
        ),
        (
            "- Sessions with relevant click lift: "
            f"{lift['sessions_with_relevant_click_rate_lift']:.2%}"
        ),
        (
            "- Avg clicks/session lift: "
            f"{lift['avg_clicks_per_session_lift']:.2%}"
        ),
        "",
        "## Interpretation",
        "",
        "Positive lift indicates the personalized policy improves expected "
        "engagement relative to the selected baseline under the simulation "
        "assumptions. Review sensitivity by changing click probabilities or "
        "using a different baseline policy.",
        "",
    ]
    return "\n".join(lines)


def evaluate_policy_impact(
    processed_dir: Path,
    artifacts_dir: Path,
    report_path: Path,
    baseline_policy: str = "popularity",
    max_train_rows: int = 0,
    max_eval_users: int = 1_000,
    n_components: int = 64,
    max_text_features: int = 20_000,
    k: int = 12,
    hybrid_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    simulation_rounds: int = 200,
    relevant_click_base: float = 0.35,
    irrelevant_click_base: float = 0.01,
    random_state: int = 42,
) -> dict[str, object]:
    """Estimate policy lift from simulated interactions on holdout users."""
    if baseline_policy not in {"popularity", "random"}:
        raise ValueError(
            "baseline_policy must be either 'popularity' or 'random'"
        )

    paths = {
        "train": processed_dir / "interactions_train.parquet",
        "validation": processed_dir / "interactions_val.parquet",
        "test": processed_dir / "interactions_test.parquet",
        "items": processed_dir / "items_clean.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `make preprocess` first."
            )

    train, _ = read_interactions(paths["train"], want_recency=False)
    validation, _ = read_interactions(paths["validation"], want_recency=False)
    test, _ = read_interactions(paths["test"], want_recency=False)
    train = _positive_limit(train, max_train_rows)
    train["customer_id"] = train["customer_id"].astype(str)
    train["article_id"] = train["article_id"].astype(str)

    items = pd.read_parquet(
        paths["items"], columns=["article_id", "item_text"]
    )

    popularity = PopularityRecommender().fit(train)
    collaborative = CollaborativeSVD(n_components, random_state).fit(train)
    content = ContentRecommender(
        n_components=n_components,
        max_features=max_text_features,
        random_state=random_state,
    ).fit(train, items, candidate_item_ids=set(train["article_id"]))

    validation_truth = build_ground_truth(
        validation,
        known_users=set(collaborative.user_ids_),
        known_items=set(collaborative.item_ids_),
        seen_by_user=popularity.seen_,
    )
    validation_users = sorted(validation_truth)
    if max_eval_users > 0 and len(validation_users) > max_eval_users:
        rng = np.random.default_rng(random_state)
        validation_users = sorted(
            rng.choice(
                validation_users,
                size=max_eval_users,
                replace=False,
            ).tolist()
        )
    validation_truth = {u: validation_truth[u] for u in validation_users}

    best_weight, tuning = _tune_hybrid_weight(
        collaborative,
        content,
        validation_truth,
        validation_users,
        k,
        hybrid_weights,
    )
    hybrid = HybridRecommender(collaborative, content, best_weight)

    # Use untouched test users to estimate policy lift after weight selection.
    truth = build_ground_truth(
        test,
        known_users=set(collaborative.user_ids_),
        known_items=set(collaborative.item_ids_),
        seen_by_user=popularity.seen_,
    )
    users = sorted(truth)
    if max_eval_users > 0 and len(users) > max_eval_users:
        rng = np.random.default_rng(random_state)
        users = sorted(
            rng.choice(users, size=max_eval_users, replace=False).tolist()
        )
    truth = {u: truth[u] for u in users}

    if baseline_policy == "popularity":
        baseline_model: object = popularity
    else:
        baseline_model = RandomPolicy(
            item_ids=[str(item) for item in collaborative.item_ids_],
            seen_by_user=popularity.seen_,
            rng=np.random.default_rng(random_state),
        )

    baseline_recs = _recommend_users(baseline_model, users, k)
    personalized_recs = _recommend_users(hybrid, users, k)

    baseline_offline = evaluate_rankings(baseline_recs, truth, k).to_dict()
    personalized_offline = evaluate_rankings(
        personalized_recs,
        truth,
        k,
    ).to_dict()

    sim_rng = np.random.default_rng(random_state)
    baseline_sim = _simulate_clicks(
        baseline_recs,
        truth,
        rounds=simulation_rounds,
        rng=sim_rng,
        relevant_click_base=relevant_click_base,
        irrelevant_click_base=irrelevant_click_base,
    ).to_dict()
    personalized_sim = _simulate_clicks(
        personalized_recs,
        truth,
        rounds=simulation_rounds,
        rng=sim_rng,
        relevant_click_base=relevant_click_base,
        irrelevant_click_base=irrelevant_click_base,
    ).to_dict()

    lift = {
        "click_through_rate_lift": _lift(
            personalized_sim["click_through_rate"],
            baseline_sim["click_through_rate"],
        ),
        "relevant_click_through_rate_lift": _lift(
            personalized_sim["relevant_click_through_rate"],
            baseline_sim["relevant_click_through_rate"],
        ),
        "sessions_with_relevant_click_rate_lift": _lift(
            personalized_sim["sessions_with_relevant_click_rate"],
            baseline_sim["sessions_with_relevant_click_rate"],
        ),
        "avg_clicks_per_session_lift": _lift(
            personalized_sim["avg_clicks_per_session"],
            baseline_sim["avg_clicks_per_session"],
        ),
    }

    report: dict[str, object] = {
        "configuration": {
            "baseline_policy": baseline_policy,
            "max_train_rows": max_train_rows,
            "max_eval_users": max_eval_users,
            "n_components": n_components,
            "max_text_features": max_text_features,
            "k": k,
            "hybrid_weights": ",".join(map(str, hybrid_weights)),
            "selected_hybrid_weight": best_weight,
            "simulation_rounds": simulation_rounds,
            "relevant_click_base": relevant_click_base,
            "irrelevant_click_base": irrelevant_click_base,
            "random_state": random_state,
        },
        "data": {
            "training_rows": len(train),
            "training_users": train["customer_id"].nunique(),
            "training_items": train["article_id"].nunique(),
            "warm_test_users": len(users),
        },
        "offline_metrics": {
            "baseline": baseline_offline,
            "personalized_hybrid": personalized_offline,
        },
        "simulated_outcomes": {
            "baseline": baseline_sim,
            "personalized_hybrid": personalized_sim,
        },
        "lift": lift,
        "hybrid_weight_tuning": tuning,
        "evaluation_scope": (
            "Hybrid weight tuned on warm validation users; policy impact "
            "simulated on untouched warm test users with position-biased "
            "click probabilities."
        ),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_dir / "policy_impact_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_markdown(report), encoding="utf-8")
    return report


def _weights(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "weights must be comma-separated numbers"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and personalized policies via simulation"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts/policy_impact")
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/policy_impact_report.md"),
    )
    parser.add_argument(
        "--baseline-policy",
        choices=("popularity", "random"),
        default="popularity",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-users", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--max-text-features", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument(
        "--hybrid-weights",
        type=_weights,
        default=(0.25, 0.5, 0.75),
    )
    parser.add_argument("--simulation-rounds", type=int, default=200)
    parser.add_argument("--relevant-click-base", type=float, default=0.35)
    parser.add_argument("--irrelevant-click-base", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    report = evaluate_policy_impact(**vars(parse_args()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
