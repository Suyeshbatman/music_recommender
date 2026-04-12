from src.recommender import (
    Song, UserProfile, Recommender,
    load_songs, score_song, recommend_songs, STRATEGIES,
)


# ── Helpers ──────────────────────────────────────────────────────

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


# ── Original Tests ───────────────────────────────────────────────

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ── Challenge 1: Advanced Features Tests ─────────────────────────

def test_load_songs_parses_advanced_features():
    songs = load_songs("data/songs.csv")
    assert len(songs) == 20
    song = songs[0]
    assert "popularity" in song
    assert "release_decade" in song
    assert "mood_tags" in song
    assert "instrumentalness" in song
    assert "liveness" in song
    assert isinstance(song["popularity"], int)
    assert isinstance(song["mood_tags"], list)
    assert len(song["mood_tags"]) > 0


def test_score_song_uses_advanced_features():
    songs = load_songs("data/songs.csv")
    prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
        "target_popularity": 85,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "uplifting"],
        "likes_instrumental": False,
        "likes_live": False,
    }
    score_with_adv, reasons_with_adv = score_song(prefs, songs[0])  # Sunrise City

    # Also score without advanced prefs
    basic_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    score_basic, _ = score_song(basic_prefs, songs[0])

    # Advanced features should add to the score
    assert score_with_adv > score_basic
    reason_text = "; ".join(reasons_with_adv)
    assert "popularity" in reason_text
    assert "decade" in reason_text
    assert "mood tags" in reason_text


def test_mood_tags_partial_match():
    song = {
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "valence": 0.8, "danceability": 0.7, "acousticness": 0.2,
        "popularity": 80, "release_decade": "2020s",
        "mood_tags": ["euphoric", "warm", "uplifting"],
        "instrumentalness": 0.1, "liveness": 0.2,
    }
    prefs = {
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "preferred_mood_tags": ["euphoric", "aggressive"],  # 1/2 match
    }
    _, reasons = score_song(prefs, song)
    reason_text = "; ".join(reasons)
    assert "mood tags" in reason_text
    assert "euphoric" in reason_text


# ── Challenge 2: Strategy Pattern Tests ──────────────────────────

def test_all_strategies_exist():
    expected = {"balanced", "genre_first", "mood_first", "energy_focused", "discovery"}
    assert set(STRATEGIES.keys()) == expected


def test_strategies_return_different_weights():
    balanced_w = STRATEGIES["balanced"].weights()
    genre_first_w = STRATEGIES["genre_first"].weights()
    assert genre_first_w["genre"] > balanced_w["genre"]
    assert balanced_w["mood"] > genre_first_w["mood"]


def test_strategy_affects_ranking():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}

    balanced_recs = recommend_songs(prefs, songs, k=3, mode="balanced")
    energy_recs = recommend_songs(prefs, songs, k=3, mode="energy_focused")

    balanced_titles = [r[0]["title"] for r in balanced_recs]
    energy_titles = [r[0]["title"] for r in energy_recs]
    # At minimum the ordering should differ for some entries
    assert balanced_titles[0] != energy_titles[0] or balanced_titles != energy_titles


def test_discovery_mode_penalizes_popularity():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "likes_acoustic": True}

    balanced_recs = recommend_songs(prefs, songs, k=5, mode="balanced")
    discovery_recs = recommend_songs(prefs, songs, k=5, mode="discovery")

    # Discovery mode should rank lower-popularity songs higher
    balanced_top_pop = balanced_recs[0][0].get("popularity", 0)
    discovery_top_pop = discovery_recs[0][0].get("popularity", 0)
    # Not necessarily lower for #1, but average popularity of top 5 should decrease
    balanced_avg = sum(r[0].get("popularity", 0) for r in balanced_recs) / 5
    discovery_avg = sum(r[0].get("popularity", 0) for r in discovery_recs) / 5
    assert discovery_avg <= balanced_avg


# ── Challenge 3: Diversity Penalty Tests ─────────────────────────

def test_diversity_penalty_reduces_artist_repeats():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "likes_acoustic": True}

    # Without diversity
    normal = recommend_songs(prefs, songs, k=5)
    normal_artists = [r[0]["artist"] for r in normal]

    # With diversity penalty
    diverse = recommend_songs(prefs, songs, k=5, diversity_penalty=1.5)
    diverse_artists = [r[0]["artist"] for r in diverse]

    # Count unique artists — diverse should have >= as many unique artists
    assert len(set(diverse_artists)) >= len(set(normal_artists))


def test_max_per_artist_hard_cap():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False}

    recs = recommend_songs(prefs, songs, k=5, max_per_artist=1)
    artists = [r[0]["artist"] for r in recs]
    # No artist should appear more than once
    assert len(artists) == len(set(artists))


def test_max_per_genre_hard_cap():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True}

    recs = recommend_songs(prefs, songs, k=5, max_per_genre=2)
    genres = [r[0]["genre"] for r in recs]
    from collections import Counter
    genre_counts = Counter(genres)
    for count in genre_counts.values():
        assert count <= 2


def test_diversity_penalty_affects_score():
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "likes_acoustic": True}

    normal = recommend_songs(prefs, songs, k=5)
    diverse = recommend_songs(prefs, songs, k=5, diversity_penalty=2.0)

    # If there were repeated artists/genres, scores should be lower in diverse mode
    normal_total = sum(r[1] for r in normal)
    diverse_total = sum(r[1] for r in diverse)
    # Diverse total can be lower or equal (never higher if penalties applied)
    assert diverse_total <= normal_total + 0.01  # small epsilon for float comparison
