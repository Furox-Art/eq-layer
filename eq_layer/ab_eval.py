from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


DIMENSIONS = (
    "intent_fidelity",
    "appropriateness",
    "actionability",
    "non_patronizing",
    "overall",
)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_blinded(
    pairs: list[dict],
    *,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Randomize baseline/EQ sides and return ballot + separate decode key."""
    rng = random.Random(seed)
    ballot: list[dict] = []
    key: dict[str, dict[str, str]] = {}

    seen: set[str] = set()
    for row in pairs:
        case_id = str(row["id"])
        if case_id in seen:
            raise ValueError(f"Duplicate A/B case id: {case_id}")
        seen.add(case_id)

        baseline = str(row["baseline"])
        eq_response = str(row["eq"])
        if not baseline.strip() or not eq_response.strip():
            raise ValueError(f"Empty response in A/B case: {case_id}")

        eq_is_a = bool(rng.getrandbits(1))
        response_a = eq_response if eq_is_a else baseline
        response_b = baseline if eq_is_a else eq_response
        key[case_id] = {
            "A": "eq" if eq_is_a else "baseline",
            "B": "baseline" if eq_is_a else "eq",
        }
        ballot.append(
            {
                "id": case_id,
                "context": row.get("context", []),
                "response_A": response_a,
                "response_B": response_b,
                "ratings": {dimension: "" for dimension in DIMENSIONS},
            }
        )

    return ballot, key


@dataclass(frozen=True)
class PreferenceSummary:
    wins: int
    losses: int
    ties: int
    win_rate_non_tie: float | None
    wilson_low: float | None
    wilson_high: float | None
    exact_p_two_sided: float | None


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0.")
    p = successes / total
    denom = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _binom_two_sided_half(successes: int, total: int) -> float:
    """Exact two-sided sign test under p=0.5, summing equally/less likely outcomes."""
    if total <= 0:
        raise ValueError("Binomial test requires total > 0.")
    observed = math.comb(total, successes)
    numerator = 0
    for k in range(total + 1):
        ways = math.comb(total, k)
        if ways <= observed:
            numerator += ways
    return min(1.0, numerator / (2 ** total))


def summarize(wins: int, losses: int, ties: int) -> PreferenceSummary:
    non_ties = wins + losses
    if non_ties == 0:
        return PreferenceSummary(
            wins=wins,
            losses=losses,
            ties=ties,
            win_rate_non_tie=None,
            wilson_low=None,
            wilson_high=None,
            exact_p_two_sided=None,
        )

    low, high = _wilson_interval(wins, non_ties)
    return PreferenceSummary(
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate_non_tie=round(wins / non_ties, 4),
        wilson_low=round(low, 4),
        wilson_high=round(high, 4),
        exact_p_two_sided=round(_binom_two_sided_half(wins, non_ties), 6),
    )


def score_blinded(rated_ballot: list[dict], key: dict) -> dict:
    counts = {
        dimension: {"wins": 0, "losses": 0, "ties": 0}
        for dimension in DIMENSIONS
    }

    for row in rated_ballot:
        case_id = str(row["id"])
        if case_id not in key:
            raise ValueError(f"Missing decode key for case: {case_id}")

        ratings = row.get("ratings") or {}
        for dimension in DIMENSIONS:
            vote = str(ratings.get(dimension, "")).strip().upper()
            if vote not in {"A", "B", "TIE"}:
                raise ValueError(
                    f"{case_id}/{dimension}: expected A, B, or tie; got {vote!r}"
                )

            if vote == "TIE":
                counts[dimension]["ties"] += 1
                continue

            winner = key[case_id][vote]
            if winner == "eq":
                counts[dimension]["wins"] += 1
            elif winner == "baseline":
                counts[dimension]["losses"] += 1
            else:
                raise ValueError(f"Invalid decode key target: {winner}")

    report = {
        "design": "blind-pairwise-human-evaluation",
        "unit": "case-level paired preference",
        "dimensions": {},
    }
    for dimension, values in counts.items():
        summary = summarize(**values)
        report["dimensions"][dimension] = {
            "eq_wins": summary.wins,
            "eq_losses": summary.losses,
            "ties": summary.ties,
            "eq_win_rate_non_tie": summary.win_rate_non_tie,
            "wilson_95_low": summary.wilson_low,
            "wilson_95_high": summary.wilson_high,
            "exact_sign_test_p_two_sided": summary.exact_p_two_sided,
        }
    return report


def _decoded_vote(row: dict, key: dict, dimension: str) -> str:
    case_id = str(row["id"])
    if case_id not in key:
        raise ValueError(f"Missing decode key for case: {case_id}")
    ratings = row.get("ratings") or {}
    vote = str(ratings.get(dimension, "")).strip().upper()
    if vote == "TIE":
        return "tie"
    if vote not in {"A", "B"}:
        raise ValueError(
            f"{case_id}/{dimension}: expected A, B, or tie; got {vote!r}"
        )
    winner = key[case_id][vote]
    if winner not in {"eq", "baseline"}:
        raise ValueError(f"Invalid decode key target: {winner}")
    return winner


def _fleiss_kappa(case_votes: list[list[str]]) -> float | None:
    if not case_votes:
        return None
    n_raters = len(case_votes[0])
    if n_raters < 2 or any(len(votes) != n_raters for votes in case_votes):
        return None

    categories = ("eq", "baseline", "tie")
    p_i = []
    totals = {category: 0 for category in categories}

    for votes in case_votes:
        counts = {category: votes.count(category) for category in categories}
        for category in categories:
            totals[category] += counts[category]
        agreement = (
            sum(count * count for count in counts.values()) - n_raters
        ) / (n_raters * (n_raters - 1))
        p_i.append(agreement)

    p_bar = sum(p_i) / len(p_i)
    denom = len(case_votes) * n_raters
    marginals = {
        category: totals[category] / denom
        for category in categories
    }
    p_e = sum(value * value for value in marginals.values())
    if math.isclose(1.0 - p_e, 0.0):
        return None
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def score_multiple_blinded(
    rated_ballots: dict[str, list[dict]],
    key: dict,
) -> dict:
    """Aggregate multiple blinded raters without treating rater-votes as cases."""
    if len(rated_ballots) < 2:
        raise ValueError("Multi-rater scoring requires at least two raters.")

    case_ids: list[str] | None = None
    rows_by_rater: dict[str, dict[str, dict]] = {}

    for rater_id, ballot in rated_ballots.items():
        mapping = {str(row["id"]): row for row in ballot}
        if len(mapping) != len(ballot):
            raise ValueError(f"Duplicate case id in rater ballot: {rater_id}")
        current_ids = sorted(mapping)
        if case_ids is None:
            case_ids = current_ids
        elif current_ids != case_ids:
            raise ValueError("All raters must score the same case ids.")
        rows_by_rater[rater_id] = mapping

    assert case_ids is not None
    report = {
        "design": "blind-pairwise-multi-rater-human-evaluation",
        "n_raters": len(rated_ballots),
        "n_cases": len(case_ids),
        "dimensions": {},
        "rater_reports": {},
    }

    for rater_id, ballot in rated_ballots.items():
        report["rater_reports"][rater_id] = score_blinded(ballot, key)

    for dimension in DIMENSIONS:
        majority_wins = 0
        majority_losses = 0
        majority_ties = 0
        per_case_votes: list[list[str]] = []

        for case_id in case_ids:
            votes = [
                _decoded_vote(rows_by_rater[rater_id][case_id], key, dimension)
                for rater_id in sorted(rows_by_rater)
            ]
            per_case_votes.append(votes)
            eq_votes = votes.count("eq")
            baseline_votes = votes.count("baseline")
            if eq_votes > baseline_votes:
                majority_wins += 1
            elif baseline_votes > eq_votes:
                majority_losses += 1
            else:
                majority_ties += 1

        summary = summarize(majority_wins, majority_losses, majority_ties)
        report["dimensions"][dimension] = {
            "case_majority_eq_wins": summary.wins,
            "case_majority_eq_losses": summary.losses,
            "case_majority_ties": summary.ties,
            "case_majority_eq_win_rate_non_tie": summary.win_rate_non_tie,
            "wilson_95_low": summary.wilson_low,
            "wilson_95_high": summary.wilson_high,
            "exact_sign_test_p_two_sided": summary.exact_p_two_sided,
            "fleiss_kappa": _fleiss_kappa(per_case_votes),
        }

    return report
