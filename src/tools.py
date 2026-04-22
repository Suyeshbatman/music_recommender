"""Tool definitions for the Claude agent.

Each tool is (a) a JSON schema that the Anthropic API understands and
(b) a Python handler that the agent dispatches to. Handlers are thin —
they forward to the existing deterministic recommender and retriever.
"""

from typing import Any, Callable, Dict, List

from src.recommender import STRATEGIES, recommend_songs
from src.retriever import Retriever

# ── Tool schemas (what Claude sees) ────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Retrieve short reference passages about genres, artists, and mood "
            "tags from the local music knowledge base. Use this FIRST to ground "
            "your understanding of what the user wants before calling "
            "recommend_songs. The knowledge base covers the exact artists and "
            "genres available in the catalog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query describing the vibe, genre, or mood the user wants.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (1-5). Default 3.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "recommend_songs",
        "description": (
            "Call the deterministic content-based recommender to get the top K "
            "songs for a given preference profile. The recommender scores every "
            "song in the 20-song catalog. ALWAYS call this to produce the "
            "actual recommendations — never invent song titles yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_prefs": {
                    "type": "object",
                    "description": (
                        "Preference profile. Keys: genre (str), mood (str), "
                        "energy (0.0-1.0), likes_acoustic (bool), target_popularity "
                        "(int 0-100, optional), preferred_decade (str like '2020s', "
                        "optional), preferred_mood_tags (list of str, optional), "
                        "likes_instrumental (bool, optional), likes_live (bool, optional)."
                    ),
                    "properties": {
                        "genre": {"type": "string"},
                        "mood": {"type": "string"},
                        "energy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "likes_acoustic": {"type": "boolean"},
                        "target_popularity": {"type": "integer", "minimum": 0, "maximum": 100},
                        "preferred_decade": {"type": "string"},
                        "preferred_mood_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "likes_instrumental": {"type": "boolean"},
                        "likes_live": {"type": "boolean"},
                    },
                    "required": ["genre", "mood", "energy", "likes_acoustic"],
                },
                "k": {
                    "type": "integer",
                    "description": "Number of recommendations to return. Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "Scoring strategy. balanced = default. genre_first = genre dominates. "
                        "mood_first = mood/mood_tags dominate. energy_focused = energy dominates. "
                        "discovery = penalizes popular tracks, rewards obscurity."
                    ),
                    "enum": list(STRATEGIES.keys()),
                },
                "diversity_penalty": {
                    "type": "number",
                    "description": "Soft penalty per repeated artist/genre in results. 0 = off.",
                    "minimum": 0.0,
                },
                "max_per_artist": {
                    "type": "integer",
                    "description": "Hard cap on songs per artist. 0 = unlimited.",
                    "minimum": 0,
                },
                "max_per_genre": {
                    "type": "integer",
                    "description": "Hard cap on songs per genre. 0 = unlimited.",
                    "minimum": 0,
                },
            },
            "required": ["user_prefs"],
        },
    },
    {
        "name": "finalize_answer",
        "description": (
            "Call this EXACTLY ONCE when you are ready to return the final "
            "recommendation to the user. Pass the list of recommended songs "
            "(use the IDs or titles from recommend_songs output) and a short "
            "natural-language explanation that ties them to the user's request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2-4 sentence natural language explanation tailored to the user's request.",
                },
                "picks": {
                    "type": "array",
                    "description": "Ordered list of song titles from the recommend_songs output.",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "picks"],
        },
    },
]


# ── Python handlers ────────────────────────────────────────────────


class ToolDispatcher:
    """Dispatches tool calls from the agent to concrete Python handlers.

    Also records the last recommend_songs result so the evaluator and
    finalize_answer handler can reference the actual songs without trusting
    the LLM to copy them verbatim.
    """

    def __init__(self, songs: List[Dict[str, Any]], retriever: Retriever):
        self.songs = songs
        self.retriever = retriever
        self.last_recommendations: List[Dict[str, Any]] = []
        self.last_user_prefs: Dict[str, Any] = {}
        self.last_mode: str = "balanced"
        self.call_count: int = 0

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(arguments)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _tool_search_knowledge_base(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        k = int(args.get("k", 3))
        results = self.retriever.search(query, k=k)
        return {
            "results": [
                {"doc_id": doc_id, "relevance": round(score, 3), "text": text}
                for doc_id, score, text in results
            ],
            "count": len(results),
        }

    def _tool_recommend_songs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        user_prefs = args.get("user_prefs", {})
        k = int(args.get("k", 5))
        mode = args.get("mode", "balanced")
        if mode not in STRATEGIES:
            mode = "balanced"
        diversity_penalty = float(args.get("diversity_penalty", 0.0))
        max_per_artist = int(args.get("max_per_artist", 0))
        max_per_genre = int(args.get("max_per_genre", 0))

        recs = recommend_songs(
            user_prefs,
            self.songs,
            k=k,
            mode=mode,
            diversity_penalty=diversity_penalty,
            max_per_artist=max_per_artist,
            max_per_genre=max_per_genre,
        )

        serialized = []
        for song, score, explanation in recs:
            serialized.append({
                "title": song.get("title"),
                "artist": song.get("artist"),
                "genre": song.get("genre"),
                "mood": song.get("mood"),
                "energy": song.get("energy"),
                "popularity": song.get("popularity"),
                "release_decade": song.get("release_decade"),
                "mood_tags": song.get("mood_tags", []),
                "score": round(float(score), 3),
                "why": explanation,
            })

        self.last_recommendations = serialized
        self.last_user_prefs = user_prefs
        self.last_mode = mode
        return {
            "mode": mode,
            "count": len(serialized),
            "recommendations": serialized,
        }

    def _tool_finalize_answer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        summary = args.get("summary", "").strip()
        picks = args.get("picks", [])
        if not self.last_recommendations:
            return {
                "error": (
                    "finalize_answer called before recommend_songs; "
                    "call recommend_songs first."
                )
            }
        return {
            "summary": summary,
            "picks": picks,
            "backing_recommendations": self.last_recommendations,
        }
