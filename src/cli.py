"""Natural-language entrypoint for the Music Recommender agent.

Usage:
    python -m src.cli "relaxing jazz for a coffee shop"
    python -m src.cli                              # interactive REPL
    python -m src.cli --trace "workout pop"        # show full step trace
"""

import argparse
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

from src.agent import AgentResult, MusicAgent
from src.logging_setup import get_logger
from src.recommender import load_songs

log = get_logger("musicrec.cli")


def _print_result(result: AgentResult, show_trace: bool = False) -> None:
    print()
    print("=" * 78)
    print(f" Query: {result.query}")
    print("=" * 78)

    if show_trace:
        print("\nAgent trace:")
        for step in result.trace:
            detail_preview = {
                k: (v if not isinstance(v, (dict, list)) or len(str(v)) < 120 else "…")
                for k, v in step.detail.items()
            }
            print(f"  [{step.step:>2}] {step.action}: {detail_preview}")

    print("\nSummary:")
    print(f"  {result.summary}")

    if result.recommendations:
        rows = []
        for i, rec in enumerate(result.recommendations, 1):
            rows.append([
                i,
                rec.get("title", ""),
                rec.get("artist", ""),
                rec.get("genre", ""),
                rec.get("popularity", ""),
                f"{rec.get('score', 0):.2f}",
            ])
        print("\nRecommendations:")
        print(tabulate(
            rows,
            headers=["#", "Title", "Artist", "Genre", "Pop", "Score"],
            tablefmt="grid",
            stralign="left",
        ))

    print(f"\nMode:          {result.mode}")
    print(f"RAG used:      {result.retrieval_used}")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Confidence:    {result.evaluation.confidence:.2f}  (ok={result.evaluation.ok})")
    if result.evaluation.issues:
        print(f"Issues:        {', '.join(result.evaluation.issues)}")
    if result.error:
        print(f"Error:         {result.error}")
    print()


def run_query(query: str, show_trace: bool = False) -> AgentResult:
    repo_root = Path(__file__).resolve().parent.parent
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    agent = MusicAgent(songs=songs)
    result = agent.run(query)
    _print_result(result, show_trace=show_trace)
    return result


def _repl(show_trace: bool) -> None:
    print("Music recommender agent — type a request, or 'quit' to exit.\n")
    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in {"quit", "exit", ":q"}:
            return
        try:
            run_query(query, show_trace=show_trace)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Music recommender agent CLI")
    parser.add_argument("query", nargs="*", help="Natural language request")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full agent step-by-step trace",
    )
    args = parser.parse_args(argv)

    if args.query:
        query = " ".join(args.query)
        try:
            run_query(query, show_trace=args.trace)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        _repl(show_trace=args.trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
