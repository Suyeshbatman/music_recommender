"""End-to-end evaluation harness (stretch feature).

Runs a fixed set of natural-language prompts through the real agent, asserts
per-prompt expectations against the returned recommendations, and prints a
summary with pass/fail counts and average confidence.

Requires ANTHROPIC_API_KEY. Skip with:
    python tests/eval_harness.py --skip-api   (runs heuristic fallback only)

Usage:
    python tests/eval_harness.py
    python tests/eval_harness.py --verbose
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

# Allow running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent import AgentResult, MusicAgent  # noqa: E402
from src.recommender import load_songs  # noqa: E402


Check = Callable[[AgentResult], bool]


def check_has_genre(genre: str) -> Check:
    def _check(result: AgentResult) -> bool:
        return any(
            (r.get("genre", "").lower() == genre.lower())
            for r in result.recommendations
        )
    _check.__name__ = f"has_genre_{genre}"
    return _check


def check_top_popularity_below(threshold: int) -> Check:
    def _check(result: AgentResult) -> bool:
        if not result.recommendations:
            return False
        top_pop = result.recommendations[0].get("popularity", 100)
        return top_pop is not None and top_pop < threshold
    _check.__name__ = f"top_pop_under_{threshold}"
    return _check


def check_avg_energy_below(threshold: float) -> Check:
    def _check(result: AgentResult) -> bool:
        energies = [r.get("energy") for r in result.recommendations if r.get("energy") is not None]
        if not energies:
            return False
        return sum(energies) / len(energies) < threshold
    _check.__name__ = f"avg_energy_under_{threshold}"
    return _check


def check_avg_energy_above(threshold: float) -> Check:
    def _check(result: AgentResult) -> bool:
        energies = [r.get("energy") for r in result.recommendations if r.get("energy") is not None]
        if not energies:
            return False
        return sum(energies) / len(energies) > threshold
    _check.__name__ = f"avg_energy_over_{threshold}"
    return _check


def check_retrieval_used() -> Check:
    def _check(result: AgentResult) -> bool:
        return result.retrieval_used
    _check.__name__ = "retrieval_used"
    return _check


def check_diverse_artists(min_unique: int) -> Check:
    def _check(result: AgentResult) -> bool:
        artists = [r.get("artist") for r in result.recommendations]
        return len(set(artists)) >= min_unique
    _check.__name__ = f"at_least_{min_unique}_unique_artists"
    return _check


EVAL_CASES: List[Dict[str, Any]] = [
    {
        "name": "lofi_study",
        "query": "I'm coding late at night and want something mellow and focused, low energy",
        "checks": [check_has_genre("lofi"), check_avg_energy_below(0.6), check_retrieval_used()],
    },
    {
        "name": "edm_workout",
        "query": "pump-up EDM for a hard workout, I want max energy",
        "checks": [check_has_genre("edm"), check_avg_energy_above(0.7), check_retrieval_used()],
    },
    {
        "name": "jazz_coffee",
        "query": "relaxing acoustic jazz for a quiet coffee shop afternoon",
        "checks": [check_has_genre("jazz"), check_avg_energy_below(0.6), check_retrieval_used()],
    },
    {
        "name": "hidden_gems",
        "query": "surface some hidden gems, nothing on the charts, off the beaten path",
        "checks": [check_top_popularity_below(70), check_retrieval_used()],
    },
    {
        "name": "diverse_rock",
        "query": "give me a variety of rock songs from different artists",
        "checks": [check_has_genre("rock"), check_diverse_artists(2), check_retrieval_used()],
    },
    {
        "name": "melancholy_mix",
        "query": "I'm feeling a bit melancholy — something bittersweet and nostalgic",
        "checks": [check_retrieval_used()],
    },
]


def run_harness(verbose: bool = False, skip_api: bool = False) -> Dict[str, Any]:
    songs = load_songs(str(REPO_ROOT / "data" / "songs.csv"))

    if skip_api:
        print("[!] --skip-api mode: using deterministic fallback for every case")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        print("[!] ANTHROPIC_API_KEY not set -- running fallback-only mode")
        skip_api = True

    agent = MusicAgent(songs=songs)

    summary = {
        "total": len(EVAL_CASES),
        "passed": 0,
        "failed": 0,
        "confidences": [],
        "per_case": [],
    }

    for case in EVAL_CASES:
        print(f"\n--- {case['name']}: {case['query']!r}")
        start = time.time()
        if skip_api:
            # Force fallback by simulating an exploding client
            class _Boom:
                class messages:
                    @staticmethod
                    def create(**kw):
                        raise RuntimeError("skip-api mode")
            agent._client = _Boom()
        try:
            result = agent.run(case["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
            summary["failed"] += 1
            summary["per_case"].append({
                "name": case["name"],
                "passed": False,
                "confidence": 0.0,
                "error": str(exc),
            })
            continue
        elapsed = time.time() - start

        check_results = []
        for check in case["checks"]:
            ok = False
            try:
                ok = check(result)
            except Exception:
                ok = False
            check_results.append((check.__name__, ok))
        all_passed = all(ok for _, ok in check_results)

        summary["confidences"].append(result.evaluation.confidence)
        if all_passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1

        print(f"  mode={result.mode}  conf={result.evaluation.confidence:.2f}  "
              f"fallback={result.fallback_used}  t={elapsed:.1f}s")
        for name, ok in check_results:
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {name}")
        if verbose:
            print(f"    picks: {result.picks}")
            if result.evaluation.issues:
                print(f"    issues: {result.evaluation.issues}")

        summary["per_case"].append({
            "name": case["name"],
            "passed": all_passed,
            "confidence": result.evaluation.confidence,
            "mode": result.mode,
            "fallback_used": result.fallback_used,
            "checks": check_results,
        })

    avg_conf = (
        sum(summary["confidences"]) / len(summary["confidences"])
        if summary["confidences"]
        else 0.0
    )
    print("\n" + "=" * 60)
    print(f" EVAL HARNESS SUMMARY")
    print("=" * 60)
    print(f"  Passed:     {summary['passed']}/{summary['total']}")
    print(f"  Failed:     {summary['failed']}/{summary['total']}")
    print(f"  Avg conf:   {avg_conf:.2f}")
    print("=" * 60)
    summary["avg_confidence"] = avg_conf
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-api", action="store_true", help="Force fallback mode")
    args = parser.parse_args(argv)
    summary = run_harness(verbose=args.verbose, skip_api=args.skip_api)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
