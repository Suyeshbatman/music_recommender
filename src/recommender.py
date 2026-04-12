import csv
import os
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Challenge 1: Advanced features
    popularity: int = 0
    release_decade: str = ""
    mood_tags: List[str] = field(default_factory=list)
    instrumentalness: float = 0.0
    liveness: float = 0.0


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Challenge 1: Advanced preference fields
    target_popularity: Optional[int] = None         # 0-100, None = no preference
    preferred_decade: Optional[str] = None          # e.g. "2020s"
    preferred_mood_tags: Optional[List[str]] = None # e.g. ["euphoric", "warm"]
    likes_instrumental: Optional[bool] = None
    likes_live: Optional[bool] = None


# ── Challenge 2: Strategy Pattern for Scoring Modes ──────────────

class ScoringStrategy:
    """Base class for scoring strategies."""
    name: str = "base"

    def weights(self) -> Dict[str, float]:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError


class BalancedStrategy(ScoringStrategy):
    """Default balanced scoring across all features."""
    name = "balanced"

    def weights(self):
        return {
            "genre": 2.0, "mood": 1.0, "energy": 1.0, "acoustic": 0.5,
            "valence": 0.3, "danceability": 0.2,
            # Advanced feature weights
            "popularity": 0.3, "decade": 0.5, "mood_tags": 0.8,
            "instrumentalness": 0.3, "liveness": 0.2,
        }

    def description(self):
        return "Balanced scoring across all features"


class GenreFirstStrategy(ScoringStrategy):
    """Heavily prioritizes genre match."""
    name = "genre_first"

    def weights(self):
        return {
            "genre": 4.0, "mood": 0.5, "energy": 0.5, "acoustic": 0.3,
            "valence": 0.2, "danceability": 0.1,
            "popularity": 0.2, "decade": 0.3, "mood_tags": 0.4,
            "instrumentalness": 0.2, "liveness": 0.1,
        }

    def description(self):
        return "Genre match dominates (4.0x weight)"


class MoodFirstStrategy(ScoringStrategy):
    """Prioritizes mood and emotional match."""
    name = "mood_first"

    def weights(self):
        return {
            "genre": 0.5, "mood": 3.0, "energy": 1.0, "acoustic": 0.5,
            "valence": 0.5, "danceability": 0.2,
            "popularity": 0.2, "decade": 0.3, "mood_tags": 1.5,
            "instrumentalness": 0.3, "liveness": 0.2,
        }

    def description(self):
        return "Mood and emotional tags dominate (3.0x + 1.5x mood_tags)"


class EnergyFocusedStrategy(ScoringStrategy):
    """Prioritizes energy and activity-based matching."""
    name = "energy_focused"

    def weights(self):
        return {
            "genre": 0.5, "mood": 0.5, "energy": 3.0, "acoustic": 0.3,
            "valence": 0.2, "danceability": 0.5,
            "popularity": 0.2, "decade": 0.2, "mood_tags": 0.4,
            "instrumentalness": 0.2, "liveness": 0.3,
        }

    def description(self):
        return "Energy similarity dominates (3.0x weight)"


class DiscoveryStrategy(ScoringStrategy):
    """Favors lesser-known songs and diverse mood tags."""
    name = "discovery"

    def weights(self):
        return {
            "genre": 1.0, "mood": 1.0, "energy": 1.0, "acoustic": 0.3,
            "valence": 0.3, "danceability": 0.2,
            "popularity": -0.5, "decade": 0.3, "mood_tags": 1.2,
            "instrumentalness": 0.3, "liveness": 0.4,
        }

    def description(self):
        return "Penalizes popular tracks, rewards mood tag overlap and liveness"


# Registry of all available strategies
STRATEGIES: Dict[str, ScoringStrategy] = {
    "balanced": BalancedStrategy(),
    "genre_first": GenreFirstStrategy(),
    "mood_first": MoodFirstStrategy(),
    "energy_focused": EnergyFocusedStrategy(),
    "discovery": DiscoveryStrategy(),
}


# ── OOP Recommender Class ───────────────────────────────────────

class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song(self, user: UserProfile, song: Song) -> float:
        """Score a single Song object against a UserProfile."""
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 2.0
        if song.mood == user.favorite_mood:
            score += 1.0
        energy_similarity = 1.0 - abs(song.energy - user.target_energy)
        score += energy_similarity
        if user.likes_acoustic:
            score += song.acousticness * 0.5
        else:
            score += (1.0 - song.acousticness) * 0.5
        score += song.valence * 0.3
        score += song.danceability * 0.2
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top k songs sorted by score descending."""
        scored = [(song, self._score_song(user, song)) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended for a user profile."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre match '{song.genre}' (+2.0)")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood match '{song.mood}' (+1.0)")
        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity {energy_sim:.2f} (+{energy_sim:.2f})")
        if user.likes_acoustic:
            reasons.append(f"acousticness {song.acousticness:.2f} (+{song.acousticness * 0.5:.2f})")
        else:
            reasons.append(f"non-acoustic bonus (+{(1.0 - song.acousticness) * 0.5:.2f})")
        reasons.append(f"valence {song.valence:.2f} (+{song.valence * 0.3:.2f})")
        reasons.append(f"danceability {song.danceability:.2f} (+{song.danceability * 0.2:.2f})")
        total = self._score_song(user, song)
        return f"Score {total:.2f}: " + ", ".join(reasons)


# ── CSV Loading ──────────────────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    if not os.path.isabs(csv_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, csv_path)
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            # Challenge 1: Parse advanced features
            row["popularity"] = int(row.get("popularity", 0))
            row["release_decade"] = row.get("release_decade", "")
            raw_tags = row.get("mood_tags", "")
            row["mood_tags"] = raw_tags.split("|") if raw_tags else []
            row["instrumentalness"] = float(row.get("instrumentalness", 0.0))
            row["liveness"] = float(row.get("liveness", 0.0))
            songs.append(row)
    return songs


# ── Scoring Function ─────────────────────────────────────────────

def score_song(user_prefs: Dict, song: Dict, mode: str = "balanced") -> Tuple[float, List[str]]:
    """Score a single song against user preferences using the selected strategy."""
    score = 0.0
    reasons = []

    # Challenge 2: Use Strategy pattern to get weights
    strategy = STRATEGIES.get(mode, STRATEGIES["balanced"])
    w = strategy.weights()

    # --- Core features ---

    # Genre match
    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += w["genre"]
        reasons.append(f"genre match (+{w['genre']:.1f})")

    # Mood match
    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += w["mood"]
        reasons.append(f"mood match (+{w['mood']:.1f})")

    # Energy similarity
    if "energy" in user_prefs and "energy" in song:
        energy_sim = 1.0 - abs(song["energy"] - user_prefs["energy"])
        energy_points = energy_sim * w["energy"]
        score += energy_points
        reasons.append(f"energy sim +{energy_points:.2f}")

    # Acoustic preference
    if "likes_acoustic" in user_prefs and "acousticness" in song:
        if user_prefs["likes_acoustic"]:
            acoustic_points = song["acousticness"] * w["acoustic"]
        else:
            acoustic_points = (1.0 - song["acousticness"]) * w["acoustic"]
        score += acoustic_points
        reasons.append(f"acoustic +{acoustic_points:.2f}")

    # Valence bonus
    if "valence" in song:
        valence_points = song["valence"] * w["valence"]
        score += valence_points
        reasons.append(f"valence +{valence_points:.2f}")

    # Danceability bonus
    if "danceability" in song:
        dance_points = song["danceability"] * w["danceability"]
        score += dance_points
        reasons.append(f"dance +{dance_points:.2f}")

    # --- Challenge 1: Advanced features ---

    # Popularity scoring (can be negative in discovery mode)
    if "target_popularity" in user_prefs and user_prefs["target_popularity"] is not None:
        pop_sim = 1.0 - abs(song.get("popularity", 50) - user_prefs["target_popularity"]) / 100.0
        pop_points = pop_sim * w.get("popularity", 0)
        score += pop_points
        reasons.append(f"popularity +{pop_points:.2f}")
    elif w.get("popularity", 0) < 0:
        # Discovery mode: penalize high-popularity songs
        pop_penalty = (song.get("popularity", 50) / 100.0) * w["popularity"]
        score += pop_penalty
        reasons.append(f"popularity {pop_penalty:+.2f}")

    # Decade match
    if user_prefs.get("preferred_decade") and song.get("release_decade"):
        if song["release_decade"] == user_prefs["preferred_decade"]:
            score += w.get("decade", 0)
            reasons.append(f"decade match +{w.get('decade', 0):.1f}")

    # Mood tags overlap (Challenge 1: detailed sub-mood matching)
    user_tags = user_prefs.get("preferred_mood_tags", None)
    song_tags = song.get("mood_tags", [])
    if user_tags and song_tags:
        matching_tags = set(t.lower() for t in user_tags) & set(t.lower() for t in song_tags)
        if matching_tags:
            tag_ratio = len(matching_tags) / len(user_tags)
            tag_points = tag_ratio * w.get("mood_tags", 0)
            score += tag_points
            reasons.append(f"mood tags [{', '.join(matching_tags)}] +{tag_points:.2f}")

    # Instrumentalness preference
    if user_prefs.get("likes_instrumental") is not None and "instrumentalness" in song:
        if user_prefs["likes_instrumental"]:
            inst_points = song["instrumentalness"] * w.get("instrumentalness", 0)
        else:
            inst_points = (1.0 - song["instrumentalness"]) * w.get("instrumentalness", 0)
        score += inst_points
        reasons.append(f"instrumental +{inst_points:.2f}")

    # Liveness preference
    if user_prefs.get("likes_live") is not None and "liveness" in song:
        if user_prefs["likes_live"]:
            live_points = song["liveness"] * w.get("liveness", 0)
        else:
            live_points = (1.0 - song["liveness"]) * w.get("liveness", 0)
        score += live_points
        reasons.append(f"liveness +{live_points:.2f}")

    return (score, reasons)


# ── Recommendation with Diversity ────────────────────────────────

def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "balanced",
    diversity_penalty: float = 0.0,
    max_per_artist: int = 0,
    max_per_genre: int = 0,
) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score and return the top k recommendations.

    Challenge 3 params:
        diversity_penalty: points deducted per duplicate artist/genre already in results
        max_per_artist: hard cap on songs per artist (0 = unlimited)
        max_per_genre: hard cap on songs per genre (0 = unlimited)
    """
    # Score every song
    scored_songs = []
    for song in songs:
        total_score, reasons = score_song(user_prefs, song, mode=mode)
        scored_songs.append((song, total_score, reasons))

    # Sort by raw score descending
    scored_songs.sort(key=lambda x: x[1], reverse=True)

    # Challenge 3: Apply diversity constraints via greedy re-ranking
    if diversity_penalty > 0 or max_per_artist > 0 or max_per_genre > 0:
        results = []
        artist_counts: Dict[str, int] = {}
        genre_counts: Dict[str, int] = {}

        for song, raw_score, reasons in scored_songs:
            artist = song.get("artist", "")
            genre = song.get("genre", "")

            # Hard caps: skip entirely if at limit
            if max_per_artist > 0 and artist_counts.get(artist, 0) >= max_per_artist:
                continue
            if max_per_genre > 0 and genre_counts.get(genre, 0) >= max_per_genre:
                continue

            # Soft penalty: reduce score for repeated artist/genre
            penalty = 0.0
            penalty_reasons = list(reasons)
            if diversity_penalty > 0:
                artist_repeats = artist_counts.get(artist, 0)
                genre_repeats = genre_counts.get(genre, 0)
                if artist_repeats > 0:
                    p = diversity_penalty * artist_repeats
                    penalty += p
                    penalty_reasons.append(f"artist repeat penalty -{p:.1f}")
                if genre_repeats > 0:
                    p = diversity_penalty * genre_repeats
                    penalty += p
                    penalty_reasons.append(f"genre repeat penalty -{p:.1f}")

            adjusted_score = raw_score - penalty
            explanation = "; ".join(penalty_reasons)
            results.append((song, adjusted_score, explanation))

            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            if len(results) >= k:
                break

        return results
    else:
        return [(song, sc, "; ".join(reasons)) for song, sc, reasons in scored_songs[:k]]
