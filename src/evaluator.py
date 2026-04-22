"""Self-check / confidence scoring for agent recommendations.

Pure function. Takes the user query, the preference profile the agent
built, and the recommendations the tool returned, and produces:

    ok:         bool — does this set of recs satisfy minimum quality bars?
    confidence: float 0.0-1.0 — how confident the agent should be
    issues:     list[str] — human-readable problems found

The confidence formula is intentionally simple and explainable:

    0.4 * genre_match_rate
  + 0.3 * avg_score_normalized
  + 0.2 * diversity_ok
  + 0.1 * retrieval_used
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# From the rubric in README.md: max balanced score is ~7.3
_MAX_SCORE = 7.3


@dataclass
class EvaluationResult:
    ok: bool
    confidence: float
    issues: List[str]
    breakdown: Dict[str, float]


def validate_results(
    query: str,
    user_prefs: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    retrieval_used: bool = False,
    diversity_requested: bool = False,
) -> EvaluationResult:
    issues: List[str] = []

    if not recommendations:
        return EvaluationResult(
            ok=False,
            confidence=0.0,
            issues=["no recommendations returned"],
            breakdown={
                "genre_match_rate": 0.0,
                "avg_score_normalized": 0.0,
                "diversity_ok": 0.0,
                "retrieval_used": 1.0 if retrieval_used else 0.0,
            },
        )

    requested_genre = (user_prefs.get("genre") or "").lower().strip()
    genre_matches = 0
    if requested_genre:
        for rec in recommendations:
            if (rec.get("genre") or "").lower().strip() == requested_genre:
                genre_matches += 1
        genre_match_rate = genre_matches / len(recommendations)
        if genre_matches == 0:
            issues.append(
                f"no result matches requested genre '{requested_genre}'"
            )
    else:
        genre_match_rate = 1.0

    scores = [float(rec.get("score", 0.0)) for rec in recommendations]
    avg_score = sum(scores) / len(scores)
    avg_score_normalized = max(0.0, min(1.0, avg_score / _MAX_SCORE))
    if avg_score < 1.0:
        issues.append(f"average score {avg_score:.2f} is very low")

    artists = [rec.get("artist") for rec in recommendations]
    unique_artists = len(set(artists))
    diversity_ratio = unique_artists / len(recommendations)
    if diversity_requested and diversity_ratio < 1.0:
        issues.append("diversity was requested but artists repeat")
        diversity_ok = diversity_ratio
    else:
        diversity_ok = 1.0 if diversity_ratio >= 0.6 else diversity_ratio

    breakdown = {
        "genre_match_rate": round(genre_match_rate, 3),
        "avg_score_normalized": round(avg_score_normalized, 3),
        "diversity_ok": round(diversity_ok, 3),
        "retrieval_used": 1.0 if retrieval_used else 0.0,
    }

    confidence = (
        0.4 * breakdown["genre_match_rate"]
        + 0.3 * breakdown["avg_score_normalized"]
        + 0.2 * breakdown["diversity_ok"]
        + 0.1 * breakdown["retrieval_used"]
    )
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    ok = confidence >= 0.5 and genre_matches > 0 if requested_genre else confidence >= 0.4

    return EvaluationResult(
        ok=ok,
        confidence=confidence,
        issues=issues,
        breakdown=breakdown,
    )
