"""Agentic workflow around the deterministic music recommender.

Flow:
    natural-language query
      -> Claude (tool use loop)
         -> search_knowledge_base  (RAG)
         -> recommend_songs        (deterministic tool)
         -> finalize_answer        (stop condition)
      -> evaluator (confidence + issues)
      -> retry once with different strategy if confidence is too low
      -> AgentResult with structured trace

Guardrails:
    * Hard cap on agent iterations (MAX_ITERATIONS)
    * Hard cap on total tool calls per session (MAX_TOOL_CALLS)
    * Any API / tool failure falls back to a deterministic balanced
      recommendation so the system always returns something.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluator import EvaluationResult, validate_results
from src.logging_setup import get_logger
from src.recommender import recommend_songs
from src.retriever import Retriever, default_retriever
from src.tools import TOOL_SCHEMAS, ToolDispatcher

log = get_logger("musicrec.agent")

MODEL_ID = "claude-haiku-4-5-20251001"
MAX_ITERATIONS = 4
MAX_TOOL_CALLS = 8
MAX_TOKENS = 1400
MIN_CONFIDENCE_BEFORE_RETRY = 0.55

SYSTEM_PROMPT = """You are a music recommendation agent for a small 20-song catalog. Your job is to take a free-form natural language music request and return a grounded, explained recommendation.

You have three tools:

1. search_knowledge_base — retrieve short reference passages about genres, artists, and mood tags from the local knowledge base. ALWAYS call this at least once FIRST to ground yourself in what is actually in the catalog. The catalog only has 20 songs across a fixed set of genres and artists — do not assume anything that is not in the knowledge base.

2. recommend_songs — run the deterministic content-based recommender. You must build a user preference profile (genre, mood, energy 0-1, likes_acoustic bool, optionally target_popularity, preferred_decade, preferred_mood_tags, likes_instrumental, likes_live) and pick a scoring mode:
   - balanced: default general case
   - genre_first: user strongly signals a specific genre
   - mood_first: user's request is about emotion / vibe more than genre
   - energy_focused: user mentions activity (workout, studying, party)
   - discovery: user asks for hidden gems, obscure, underground, off-the-beaten-path

   You can also set diversity_penalty (try 1.0 if you want variety) or max_per_artist=1 for variety.

3. finalize_answer — call this EXACTLY ONCE when done, with a short natural-language summary and the ordered list of song titles.

Rules:
- Never invent song titles, artists, or scores. Only use what recommend_songs returns.
- Always call search_knowledge_base before recommend_songs.
- Call finalize_answer exactly once to end the turn.
- Keep your summary 2-4 sentences and tie it to the user's actual wording."""


@dataclass
class TraceStep:
    step: int
    action: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    query: str
    summary: str
    picks: List[str]
    recommendations: List[Dict[str, Any]]
    user_prefs: Dict[str, Any]
    mode: str
    retrieval_used: bool
    evaluation: EvaluationResult
    trace: List[TraceStep]
    fallback_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "picks": self.picks,
            "mode": self.mode,
            "retrieval_used": self.retrieval_used,
            "fallback_used": self.fallback_used,
            "confidence": self.evaluation.confidence,
            "ok": self.evaluation.ok,
            "issues": self.evaluation.issues,
            "confidence_breakdown": self.evaluation.breakdown,
            "user_prefs": self.user_prefs,
            "recommendations": self.recommendations,
            "trace": [
                {"step": s.step, "action": s.action, **s.detail}
                for s in self.trace
            ],
            "error": self.error,
        }


class MusicAgent:
    def __init__(
        self,
        songs: List[Dict[str, Any]],
        retriever: Optional[Retriever] = None,
        client: Any = None,
    ):
        self.songs = songs
        self.retriever = retriever or default_retriever()
        self._client = client
        self._dispatcher: Optional[ToolDispatcher] = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set."
            )
        self._client = anthropic.Anthropic()
        return self._client

    # ── Public entrypoint ──────────────────────────────────────────

    def run(self, query: str) -> AgentResult:
        self._dispatcher = ToolDispatcher(self.songs, self.retriever)
        trace: List[TraceStep] = []
        trace.append(TraceStep(step=0, action="user_query", detail={"query": query}))
        log.info("agent.run query=%r", query)

        try:
            final_payload = self._run_tool_loop(query, trace)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent loop failed: %s", _short_error(exc))
            short = _short_error(exc)
            trace.append(
                TraceStep(
                    step=len(trace),
                    action="error",
                    detail={"message": short},
                )
            )
            return self._fallback(query, trace, error=short)

        summary = final_payload.get("summary", "")
        picks = final_payload.get("picks", [])
        recs = self._dispatcher.last_recommendations
        user_prefs = self._dispatcher.last_user_prefs
        mode = self._dispatcher.last_mode
        retrieval_used = any(s.action == "tool:search_knowledge_base" for s in trace)
        diversity_requested = bool(
            user_prefs.get("_diversity_requested", False)
        )

        evaluation = validate_results(
            query=query,
            user_prefs=user_prefs,
            recommendations=recs,
            retrieval_used=retrieval_used,
            diversity_requested=diversity_requested,
        )
        trace.append(
            TraceStep(
                step=len(trace),
                action="evaluate",
                detail={
                    "confidence": evaluation.confidence,
                    "ok": evaluation.ok,
                    "issues": evaluation.issues,
                    "breakdown": evaluation.breakdown,
                },
            )
        )
        log.info(
            "agent.evaluate confidence=%.2f ok=%s issues=%s",
            evaluation.confidence,
            evaluation.ok,
            evaluation.issues,
        )

        # Retry-once with a different strategy if confidence is too low
        if (
            evaluation.confidence < MIN_CONFIDENCE_BEFORE_RETRY
            and recs
            and mode != "balanced"
        ):
            log.info("agent.retry switching to balanced mode")
            trace.append(
                TraceStep(
                    step=len(trace),
                    action="retry",
                    detail={"reason": "low confidence", "new_mode": "balanced"},
                )
            )
            retry_recs = recommend_songs(
                user_prefs, self.songs, k=5, mode="balanced"
            )
            serialized_retry = [
                {
                    "title": s.get("title"),
                    "artist": s.get("artist"),
                    "genre": s.get("genre"),
                    "score": round(float(score), 3),
                    "why": explanation,
                }
                for s, score, explanation in retry_recs
            ]
            retry_eval = validate_results(
                query=query,
                user_prefs=user_prefs,
                recommendations=serialized_retry,
                retrieval_used=retrieval_used,
            )
            if retry_eval.confidence > evaluation.confidence:
                recs = serialized_retry
                evaluation = retry_eval
                mode = "balanced"
                picks = [r["title"] for r in serialized_retry]
                summary = (
                    summary
                    + " (Note: switched to balanced strategy after self-check.)"
                )

        return AgentResult(
            query=query,
            summary=summary,
            picks=picks,
            recommendations=recs,
            user_prefs=user_prefs,
            mode=mode,
            retrieval_used=retrieval_used,
            evaluation=evaluation,
            trace=trace,
            fallback_used=False,
        )

    # ── Tool-use loop ──────────────────────────────────────────────

    def _run_tool_loop(
        self, query: str, trace: List[TraceStep]
    ) -> Dict[str, Any]:
        client = self._get_client()
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": query},
        ]

        system_blocks = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        final_payload: Optional[Dict[str, Any]] = None

        for iteration in range(MAX_ITERATIONS):
            log.info("agent.loop iteration=%d", iteration)
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=system_blocks,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            assistant_content = _extract_content_blocks(response)
            messages.append({"role": "assistant", "content": assistant_content})

            stop_reason = _get_stop_reason(response)
            tool_uses = [b for b in assistant_content if _block_type(b) == "tool_use"]

            # Record any text the assistant emitted
            for block in assistant_content:
                if _block_type(block) == "text":
                    text = _block_text(block).strip()
                    if text:
                        trace.append(
                            TraceStep(
                                step=len(trace),
                                action="assistant_text",
                                detail={"text": text[:500]},
                            )
                        )

            if not tool_uses:
                break

            tool_results_content: List[Dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_name = _block_get(tool_use, "name")
                tool_input = _block_get(tool_use, "input") or {}
                tool_id = _block_get(tool_use, "id")

                if self._dispatcher.call_count >= MAX_TOOL_CALLS:
                    result = {"error": "tool call budget exhausted"}
                else:
                    log.info(
                        "agent.tool_call name=%s input=%s",
                        tool_name,
                        json.dumps(tool_input)[:300],
                    )
                    result = self._dispatcher.dispatch(tool_name, tool_input)

                trace.append(
                    TraceStep(
                        step=len(trace),
                        action=f"tool:{tool_name}",
                        detail={
                            "input": tool_input,
                            "result_summary": _summarize_tool_result(tool_name, result),
                        },
                    )
                )

                if tool_name == "finalize_answer" and "error" not in result:
                    final_payload = result

                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            messages.append({"role": "user", "content": tool_results_content})

            if final_payload is not None:
                break

            if stop_reason == "end_turn":
                break

        if final_payload is None:
            raise RuntimeError(
                "agent did not call finalize_answer within iteration budget"
            )
        return final_payload

    # ── Fallback ───────────────────────────────────────────────────

    def _fallback(
        self, query: str, trace: List[TraceStep], error: str
    ) -> AgentResult:
        """Deterministic fallback — used when the agent loop throws."""
        log.warning("using deterministic fallback: %s", error)
        heuristic_prefs = _heuristic_prefs_from_query(query)

        # RAG still works without the LLM — run the retriever to enrich prefs
        retrieval_used = False
        rag_context = ""
        try:
            rag_results = self.retriever.search(query, k=3)
            if rag_results:
                retrieval_used = True
                rag_context = self.retriever.format_context(rag_results)
                trace.append(
                    TraceStep(
                        step=len(trace),
                        action="tool:search_knowledge_base",
                        detail={
                            "input": {"query": query, "k": 3},
                            "result_summary": {
                                "count": len(rag_results),
                                "doc_ids": [r[0] for r in rag_results],
                            },
                        },
                    )
                )
                heuristic_prefs = _enrich_prefs_from_rag(
                    heuristic_prefs, rag_results
                )
        except Exception:  # noqa: BLE001
            pass

        mode = _pick_mode_from_query(query)
        recs = recommend_songs(heuristic_prefs, self.songs, k=5, mode=mode)
        serialized = [
            {
                "title": s.get("title"),
                "artist": s.get("artist"),
                "genre": s.get("genre"),
                "mood": s.get("mood"),
                "energy": s.get("energy"),
                "popularity": s.get("popularity"),
                "release_decade": s.get("release_decade"),
                "mood_tags": s.get("mood_tags", []),
                "score": round(float(score), 3),
                "why": explanation,
            }
            for s, score, explanation in recs
        ]
        evaluation = validate_results(
            query=query,
            user_prefs=heuristic_prefs,
            recommendations=serialized,
            retrieval_used=retrieval_used,
        )
        trace.append(
            TraceStep(
                step=len(trace),
                action="fallback",
                detail={"reason": error, "mode": mode},
            )
        )
        summary = (
            "LLM unavailable -- using RAG-enhanced fallback with heuristic "
            "query parsing."
            if retrieval_used
            else "Agent backend unavailable -- returning a deterministic "
            "recommendation based on a best-effort parse of your query."
        )
        return AgentResult(
            query=query,
            summary=summary,
            picks=[r["title"] for r in serialized],
            recommendations=serialized,
            user_prefs=heuristic_prefs,
            mode=mode,
            retrieval_used=retrieval_used,
            evaluation=evaluation,
            trace=trace,
            fallback_used=True,
            error=error,
        )


# ── Response parsing helpers (duck-typed so tests can pass dicts) ──


def _extract_content_blocks(response: Any) -> List[Any]:
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    return list(content or [])


def _get_stop_reason(response: Any) -> Optional[str]:
    if hasattr(response, "stop_reason"):
        return response.stop_reason
    if isinstance(response, dict):
        return response.get("stop_reason")
    return None


def _block_type(block: Any) -> str:
    if hasattr(block, "type"):
        return block.type
    if isinstance(block, dict):
        return block.get("type", "")
    return ""


def _block_text(block: Any) -> str:
    if hasattr(block, "text"):
        return block.text or ""
    if isinstance(block, dict):
        return block.get("text", "") or ""
    return ""


def _block_get(block: Any, key: str) -> Any:
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, dict):
        return block.get(key)
    return None


def _short_error(exc: Exception) -> str:
    msg = str(exc)
    if len(msg) > 200:
        msg = msg[:197] + "..."
    return msg


def _summarize_tool_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"error": result["error"]}
    if name == "search_knowledge_base":
        return {
            "count": result.get("count", 0),
            "doc_ids": [r.get("doc_id") for r in result.get("results", [])],
        }
    if name == "recommend_songs":
        return {
            "mode": result.get("mode"),
            "count": result.get("count"),
            "top_title": (
                result.get("recommendations", [{}])[0].get("title")
                if result.get("recommendations")
                else None
            ),
        }
    if name == "finalize_answer":
        return {"picks_count": len(result.get("picks", []))}
    return {"keys": list(result.keys())}


# ── Heuristic fallback parser ──────────────────────────────────────


_GENRE_KEYWORDS = {
    "lofi": ["lofi", "lo-fi", "study", "focus", "coding"],
    "jazz": ["jazz", "coffee shop", "mellow"],
    "edm": ["edm", "rave", "festival", "party", "drop"],
    "rock": ["rock", "guitar"],
    "metal": ["metal", "heavy"],
    "pop": ["pop", "top 40", "radio"],
    "ambient": ["ambient", "sleep", "meditate", "background"],
    "country": ["country"],
    "folk": ["folk"],
    "r&b": ["r&b", "rnb", "soul"],
    "latin": ["latin", "salsa"],
    "synthwave": ["synthwave", "retrowave", "80s"],
    "indie pop": ["indie"],
}

_MOOD_KEYWORDS = {
    "chill": ["chill", "relax", "calm"],
    "happy": ["happy", "upbeat", "sunny"],
    "melancholy": ["sad", "melancholy", "down", "blue"],
    "intense": ["intense", "pump", "hard", "workout", "gym"],
    "energetic": ["energetic", "party", "dance", "hype"],
    "relaxed": ["relaxed", "coffee", "evening"],
    "nostalgic": ["nostalgic", "memories"],
    "romantic": ["romantic", "date", "sensual"],
    "moody": ["moody", "dark", "brooding"],
}


def _heuristic_prefs_from_query(query: str) -> Dict[str, Any]:
    q = query.lower()
    genre = "pop"
    for g, kws in _GENRE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            genre = g
            break
    mood = "chill"
    for m, kws in _MOOD_KEYWORDS.items():
        if any(kw in q for kw in kws):
            mood = m
            break
    if any(w in q for w in ["workout", "gym", "pump", "party", "rave"]):
        energy = 0.9
    elif any(w in q for w in ["study", "focus", "sleep", "relax", "calm"]):
        energy = 0.35
    else:
        energy = 0.6
    likes_acoustic = any(w in q for w in ["acoustic", "coffee", "folk", "jazz"])
    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }


def _pick_mode_from_query(query: str) -> str:
    """Choose a scoring strategy based on keywords in the query."""
    q = query.lower()
    if any(w in q for w in ["hidden", "obscure", "underground", "deep cut", "gem"]):
        return "discovery"
    if any(w in q for w in ["workout", "gym", "pump", "run", "exercise"]):
        return "energy_focused"
    if any(w in q for w in ["sad", "melancholy", "vibe", "mood", "feeling", "bittersweet"]):
        return "mood_first"
    return "balanced"


def _enrich_prefs_from_rag(
    prefs: Dict[str, Any],
    rag_results: list,
) -> Dict[str, Any]:
    """Use retrieved knowledge-base text to add mood_tags to heuristic prefs."""
    prefs = dict(prefs)
    _TAG_POOL = [
        "euphoric", "uplifting", "mellow", "dreamy", "aggressive", "powerful",
        "peaceful", "ethereal", "warm", "nostalgic", "bittersweet", "brooding",
        "focused", "sensual",
    ]
    combined_text = " ".join(text.lower() for _, _, text in rag_results)
    matched_tags = [t for t in _TAG_POOL if t in combined_text]
    if matched_tags:
        prefs["preferred_mood_tags"] = matched_tags[:3]
    return prefs
