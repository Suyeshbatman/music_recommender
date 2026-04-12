"""
Command line runner for the Music Recommender Simulation.

Demonstrates all features including optional extensions:
  - Challenge 1: Advanced song features (popularity, decade, mood tags, etc.)
  - Challenge 2: Multiple scoring modes via Strategy pattern
  - Challenge 3: Diversity penalty to prevent artist/genre repetition
  - Challenge 4: Formatted tables using tabulate
"""

from tabulate import tabulate
from recommender import load_songs, recommend_songs, STRATEGIES


# ── User Profiles ────────────────────────────────────────────────

PROFILES = {
    "High-Energy Pop Fan": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "likes_acoustic": False,
        "target_popularity": 80,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "uplifting"],
        "likes_instrumental": False,
        "likes_live": False,
    },
    "Chill Lofi Listener": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
        "target_popularity": 55,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["dreamy", "peaceful", "mellow"],
        "likes_instrumental": True,
        "likes_live": False,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
        "target_popularity": 75,
        "preferred_decade": "2010s",
        "preferred_mood_tags": ["aggressive", "powerful"],
        "likes_instrumental": False,
        "likes_live": True,
    },
    "Mellow Jazz Lover": {
        "genre": "jazz",
        "mood": "relaxed",
        "energy": 0.40,
        "likes_acoustic": True,
        "target_popularity": 50,
        "preferred_decade": "2000s",
        "preferred_mood_tags": ["warm", "nostalgic", "mellow"],
        "likes_instrumental": False,
        "likes_live": False,
    },
    "EDM Party Mode": {
        "genre": "edm",
        "mood": "energetic",
        "energy": 0.95,
        "likes_acoustic": False,
        "target_popularity": 85,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "aggressive"],
        "likes_instrumental": True,
        "likes_live": True,
    },
    # Adversarial / edge-case profiles
    "Conflicting (high energy + sad)": {
        "genre": "pop",
        "mood": "melancholy",
        "energy": 0.90,
        "likes_acoustic": False,
        "preferred_mood_tags": ["bittersweet", "brooding"],
    },
    "Mid Everything": {
        "genre": "indie pop",
        "mood": "nostalgic",
        "energy": 0.50,
        "likes_acoustic": False,
        "preferred_mood_tags": ["warm", "nostalgic"],
    },
}


def section_header(title: str, subtitle: str = ""):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 80)


def print_profile_info(name: str, prefs: dict):
    """Print user profile details."""
    core = f"genre={prefs['genre']}, mood={prefs['mood']}, energy={prefs['energy']}"
    tags = prefs.get("preferred_mood_tags", [])
    extras = []
    if prefs.get("target_popularity") is not None:
        extras.append(f"pop={prefs['target_popularity']}")
    if prefs.get("preferred_decade"):
        extras.append(f"decade={prefs['preferred_decade']}")
    if tags:
        extras.append(f"tags=[{', '.join(tags)}]")
    print(f"\n  Profile: {name}")
    print(f"  Core:    {core}")
    if extras:
        print(f"  Adv:     {', '.join(extras)}")


def format_reasons(explanation: str, max_width: int = 40) -> str:
    """Wrap a long reasons string for table display."""
    parts = explanation.split("; ")
    lines = []
    current = ""
    for p in parts:
        if current and len(current) + len(p) + 2 > max_width:
            lines.append(current)
            current = p
        else:
            current = current + "; " + p if current else p
    if current:
        lines.append(current)
    return "\n".join(lines)


def print_recommendations_table(
    profile_name, user_prefs, songs, mode="balanced", k=5,
    diversity_penalty=0.0, max_per_artist=0, max_per_genre=0,
):
    """Print recommendations as a formatted tabulate table (Challenge 4)."""
    print_profile_info(profile_name, user_prefs)
    strategy = STRATEGIES.get(mode, STRATEGIES["balanced"])
    mode_label = f"  Mode:    {strategy.name} - {strategy.description()}"
    if diversity_penalty > 0 or max_per_artist > 0 or max_per_genre > 0:
        div_parts = []
        if diversity_penalty > 0:
            div_parts.append(f"penalty={diversity_penalty}")
        if max_per_artist > 0:
            div_parts.append(f"max/artist={max_per_artist}")
        if max_per_genre > 0:
            div_parts.append(f"max/genre={max_per_genre}")
        mode_label += f"  |  Diversity: {', '.join(div_parts)}"
    print(mode_label)

    recs = recommend_songs(
        user_prefs, songs, k=k, mode=mode,
        diversity_penalty=diversity_penalty,
        max_per_artist=max_per_artist,
        max_per_genre=max_per_genre,
    )

    rows = []
    for i, (song, score, explanation) in enumerate(recs, 1):
        tags_str = ", ".join(song.get("mood_tags", []))
        rows.append([
            i,
            song["title"],
            song["artist"],
            song["genre"],
            song.get("popularity", ""),
            f"{score:.2f}",
            format_reasons(explanation),
        ])

    headers = ["#", "Title", "Artist", "Genre", "Pop", "Score", "Reasons"]
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))
    print()


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"\nLoaded {len(songs)} songs with {len(songs[0])} features each.")

    # ── Section 1: All profiles, balanced mode ───────────────────
    section_header("BALANCED MODE RECOMMENDATIONS",
                   "Core + advanced features, default weights")
    for name, prefs in PROFILES.items():
        print_recommendations_table(name, prefs, songs)

    # ── Section 2: Strategy pattern demo (Challenge 2) ───────────
    section_header("CHALLENGE 2: SCORING STRATEGY COMPARISON",
                   "Same profile, different strategies")
    test_profile = "High-Energy Pop Fan"
    for mode_name in STRATEGIES:
        print_recommendations_table(test_profile, PROFILES[test_profile], songs, mode=mode_name)

    # ── Section 3: Diversity penalty demo (Challenge 3) ──────────
    section_header("CHALLENGE 3: DIVERSITY PENALTY",
                   "Preventing artist/genre repetition in top results")

    # Without diversity
    print("\n  --- WITHOUT diversity penalty ---")
    print_recommendations_table(
        "Mellow Jazz Lover", PROFILES["Mellow Jazz Lover"], songs, k=5,
    )

    # With soft diversity penalty
    print("  --- WITH diversity penalty (1.0 per repeat) ---")
    print_recommendations_table(
        "Mellow Jazz Lover", PROFILES["Mellow Jazz Lover"], songs, k=5,
        diversity_penalty=1.0,
    )

    # With hard caps
    print("  --- WITH hard caps (max 1 per artist, max 2 per genre) ---")
    print_recommendations_table(
        "Mellow Jazz Lover", PROFILES["Mellow Jazz Lover"], songs, k=5,
        max_per_artist=1, max_per_genre=2,
    )

    # ── Section 4: Discovery mode (Challenge 1 advanced) ─────────
    section_header("CHALLENGE 1: DISCOVERY MODE",
                   "Penalizes popular tracks, surfaces hidden gems")
    print_recommendations_table(
        "Chill Lofi Listener", PROFILES["Chill Lofi Listener"], songs,
        mode="discovery", k=5,
    )
    print_recommendations_table(
        "Deep Intense Rock", PROFILES["Deep Intense Rock"], songs,
        mode="discovery", k=5,
    )

    # ── Section 5: Experiments (weight shifts) ───────────────────
    section_header("EXPERIMENTS: WEIGHT SENSITIVITY",
                   "How different modes change the same profile's results")

    for profile_name in ["Conflicting (high energy + sad)", "EDM Party Mode"]:
        for mode in ["balanced", "mood_first", "energy_focused"]:
            print_recommendations_table(profile_name, PROFILES[profile_name], songs, mode=mode)


if __name__ == "__main__":
    main()
