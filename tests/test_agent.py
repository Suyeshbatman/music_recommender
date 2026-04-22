"""Agent tests with a mocked Anthropic client.

These tests never hit the network — they simulate the Messages API
with a fake client that returns a queued sequence of responses.
"""

from typing import Any, Dict, List

import pytest

from src.agent import MusicAgent, _heuristic_prefs_from_query
from src.evaluator import validate_results
from src.recommender import load_songs
from src.tools import ToolDispatcher
from src.retriever import default_retriever


# ── Fake Anthropic client ──────────────────────────────────────────


class FakeResponse:
    def __init__(self, content: List[Dict[str, Any]], stop_reason: str = "tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, responses: List[FakeResponse]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("FakeMessages ran out of scripted responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: List[FakeResponse]):
        self.messages = FakeMessages(responses)


# ── Helpers ────────────────────────────────────────────────────────


def _tool_use_block(name: str, tool_id: str, inp: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _make_agent(responses: List[FakeResponse]) -> MusicAgent:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    return MusicAgent(
        songs=songs,
        retriever=default_retriever(),
        client=FakeClient(responses),
    )


# ── Tests ──────────────────────────────────────────────────────────


def test_agent_happy_path_three_tool_calls():
    responses = [
        FakeResponse(
            content=[
                _text_block("Let me check the knowledge base."),
                _tool_use_block(
                    "search_knowledge_base",
                    "t1",
                    {"query": "high energy pop workout", "k": 2},
                ),
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[
                _tool_use_block(
                    "recommend_songs",
                    "t2",
                    {
                        "user_prefs": {
                            "genre": "pop",
                            "mood": "happy",
                            "energy": 0.9,
                            "likes_acoustic": False,
                        },
                        "k": 5,
                        "mode": "energy_focused",
                    },
                )
            ],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[
                _tool_use_block(
                    "finalize_answer",
                    "t3",
                    {
                        "summary": "High-energy pop picks for your workout.",
                        "picks": ["Gym Hero", "Sunrise City"],
                    },
                )
            ],
            stop_reason="tool_use",
        ),
    ]
    agent = _make_agent(responses)
    result = agent.run("workout pop high energy")
    assert not result.fallback_used
    assert result.summary.startswith("High-energy pop")
    assert len(result.recommendations) == 5
    assert result.retrieval_used is True
    assert result.mode in {"energy_focused", "balanced"}
    assert any(s.action == "tool:search_knowledge_base" for s in result.trace)
    assert any(s.action == "tool:recommend_songs" for s in result.trace)
    assert any(s.action == "evaluate" for s in result.trace)


def test_agent_fallback_when_api_raises():
    class ExplodingClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("simulated API outage")

    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    agent = MusicAgent(
        songs=songs,
        retriever=default_retriever(),
        client=ExplodingClient(),
    )
    result = agent.run("chill lofi for studying")
    assert result.fallback_used is True
    assert result.error is not None
    assert len(result.recommendations) > 0
    # Heuristic should catch lofi
    assert result.user_prefs["genre"] == "lofi"


def test_agent_fails_cleanly_if_finalize_not_called():
    responses = [
        FakeResponse(
            content=[
                _tool_use_block(
                    "search_knowledge_base", "t1", {"query": "pop"}
                )
            ],
            stop_reason="tool_use",
        ),
        # Returns only text, never calls finalize — loop should exit
        FakeResponse(
            content=[_text_block("I have thoughts but no tool call.")],
            stop_reason="end_turn",
        ),
    ]
    agent = _make_agent(responses)
    result = agent.run("anything")
    assert result.fallback_used is True
    assert "finalize" in (result.error or "").lower()


def test_tool_call_budget_guardrail():
    """If the LLM keeps calling tools forever, we should cap it."""
    responses = [
        FakeResponse(
            content=[
                _tool_use_block(
                    "search_knowledge_base", f"t{i}", {"query": "pop"}
                )
            ],
            stop_reason="tool_use",
        )
        for i in range(12)
    ]
    agent = _make_agent(responses)
    result = agent.run("spam")
    # Either iteration cap or tool call budget stops it, then fallback
    assert result.fallback_used is True


def test_heuristic_prefs_extracts_genre_and_mood():
    prefs = _heuristic_prefs_from_query("I want some chill lofi for studying late at night")
    assert prefs["genre"] == "lofi"
    assert prefs["energy"] < 0.5
    assert prefs["mood"] in {"chill", "nostalgic", "relaxed"}


def test_heuristic_prefs_workout():
    prefs = _heuristic_prefs_from_query("give me pump up EDM for a workout")
    assert prefs["energy"] >= 0.8
    assert prefs["genre"] in {"edm", "pop", "rock"}


def test_tool_dispatcher_recommend_songs_records_state():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    disp = ToolDispatcher(songs, default_retriever())
    result = disp.dispatch(
        "recommend_songs",
        {
            "user_prefs": {
                "genre": "jazz",
                "mood": "relaxed",
                "energy": 0.4,
                "likes_acoustic": True,
            },
            "k": 3,
        },
    )
    assert result["count"] == 3
    assert disp.last_mode == "balanced"
    assert disp.last_user_prefs["genre"] == "jazz"
    assert len(disp.last_recommendations) == 3


def test_tool_dispatcher_unknown_tool_returns_error():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    songs = load_songs(str(repo_root / "data" / "songs.csv"))
    disp = ToolDispatcher(songs, default_retriever())
    result = disp.dispatch("nonsense_tool", {})
    assert "error" in result


def test_evaluator_flags_missing_genre():
    prefs = {"genre": "metal"}
    recs = [
        {"title": "a", "artist": "A", "genre": "pop", "score": 3.0},
        {"title": "b", "artist": "B", "genre": "pop", "score": 3.0},
    ]
    ev = validate_results("want metal", prefs, recs)
    assert ev.ok is False
    assert any("metal" in issue for issue in ev.issues)


def test_evaluator_high_confidence_when_everything_matches():
    prefs = {"genre": "lofi"}
    recs = [
        {"title": f"t{i}", "artist": f"A{i}", "genre": "lofi", "score": 5.0}
        for i in range(5)
    ]
    ev = validate_results("lofi please", prefs, recs, retrieval_used=True)
    assert ev.confidence > 0.7
    assert ev.ok is True
