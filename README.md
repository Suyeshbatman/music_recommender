# Music Recommender Simulation

## Project Summary

This project is a content-based music recommender system that suggests songs from a small catalog based on a user's preferred genre, mood, energy level, acoustic taste, and advanced features like popularity, release decade, and detailed mood tags. It uses a weighted scoring algorithm to rank every song in the dataset and return the top matches, along with transparent explanations of why each song was recommended. The system supports multiple scoring modes via a Strategy pattern (balanced, genre-first, mood-first, energy-focused, discovery) and includes a diversity penalty system to prevent artist/genre repetition in results.

---

## How The System Works

### How Real-World Recommendations Work

Major streaming platforms like Spotify and YouTube use two main approaches:

- **Collaborative filtering** looks at what similar users listened to. If User A and User B both love the same 50 songs, and User A also loves Song X, the system guesses User B will like Song X too. This approach doesn't need to know anything about the songs themselves. It relies entirely on patterns in user behavior (plays, skips, saves, playlist adds).

- **Content-based filtering** looks at the attributes of the songs themselves, genre, tempo, energy, mood, acousticness, etc. If a user likes high-energy pop songs, the system finds other songs with similar audio features. This is what our simulation implements.

Real platforms combine both approaches (hybrid filtering) and also use deep learning on audio signals, natural language processing on lyrics, and contextual signals like time of day or device type.

### What Our Version Prioritizes

Our system is a **pure content-based recommender**. It compares song attributes against a user's stated preferences using a math-based scoring rule.

### Features Used

Each **Song** object has these attributes:
- `genre` — categorical (pop, rock, lofi, jazz, edm, etc.)
- `mood` — categorical (happy, chill, intense, relaxed, energetic, etc.)
- `energy` — numerical, 0.0 to 1.0
- `valence` — numerical, 0.0 to 1.0 (musical positiveness)
- `danceability` — numerical, 0.0 to 1.0
- `acousticness` — numerical, 0.0 to 1.0
- `tempo_bpm` — numerical (beats per minute)
- `popularity` — integer, 0–100 (Challenge 1)
- `release_decade` — categorical, e.g. "2020s" (Challenge 1)
- `mood_tags` — list of detailed sub-moods, e.g. ["euphoric", "uplifting"] (Challenge 1)
- `instrumentalness` — numerical, 0.0 to 1.0 (Challenge 1)
- `liveness` — numerical, 0.0 to 1.0 (Challenge 1)

Each **UserProfile** stores:
- `favorite_genre` — the user's preferred genre
- `favorite_mood` — the user's preferred mood
- `target_energy` — the user's ideal energy level (0.0–1.0)
- `likes_acoustic` — boolean preference for acoustic vs. electronic sound
- `target_popularity` — preferred popularity range, 0–100 (Challenge 1)
- `preferred_decade` — e.g. "2020s" (Challenge 1)
- `preferred_mood_tags` — list of desired sub-moods (Challenge 1)
- `likes_instrumental` — boolean preference for instrumental tracks (Challenge 1)
- `likes_live` — boolean preference for live-sounding tracks (Challenge 1)

### Algorithm Recipe (Balanced Mode)

For each song in the catalog, the system calculates a score:

| Feature | Rule | Points |
|---------|------|--------|
| Genre | Exact match with user's favorite | +2.0 |
| Mood | Exact match with user's preferred mood | +1.0 |
| Energy | `(1.0 - abs(song_energy - user_target)) * weight` | up to +1.0 |
| Acousticness | If user likes acoustic: `acousticness * 0.5`; else: `(1 - acousticness) * 0.5` | up to +0.5 |
| Valence | `valence * 0.3` | up to +0.3 |
| Danceability | `danceability * 0.2` | up to +0.2 |
| Popularity | `(1 - abs(song_pop - target_pop) / 100) * 0.3` | up to +0.3 |
| Decade | Exact match with preferred decade | +0.5 |
| Mood Tags | `(matching_tags / total_tags) * 0.8` | up to +0.8 |
| Instrumentalness | Based on preference: `value * 0.3` or `(1 - value) * 0.3` | up to +0.3 |
| Liveness | Based on preference: `value * 0.2` or `(1 - value) * 0.2` | up to +0.2 |

**Maximum possible score: ~7.3** (perfect match on all core + advanced features).

The **Scoring Rule** judges one song. The **Ranking Rule** sorts all scored songs from highest to lowest and returns the top K.

### Why We Need Both a Scoring Rule and a Ranking Rule

- The **Scoring Rule** answers: "How good is this one song for this user?" It produces a single number.
- The **Ranking Rule** answers: "Of all the songs, which are the best?" It takes the output of every scoring rule call and sorts them. Without scoring, we can't compare songs. Without ranking, we can't pick the best ones.

### Data Flow

```
Input (User Prefs)
      |
      v
Process (Loop through every song in CSV)
      |
      v
For each song: apply Scoring Rule -> (score, reasons)
      |
      v
Sort all (song, score) pairs by score descending
      |
      v
Output: Top K Recommendations with explanations
```

### Potential Biases

- The system may over-prioritize genre because it carries the highest weight (+2.0), meaning a perfect energy match from a different genre can't outscore a mediocre same-genre song.
- Categorical matching is binary (match or no match). "indie pop" and "pop" are treated as completely different genres.
- Without the diversity penalty enabled, the system may recommend very similar songs from the same artist.

---

## Optional Extensions Implemented

### Challenge 1: Advanced Song Features
Added 5 new attributes to every song: `popularity` (0-100), `release_decade`, `mood_tags` (detailed sub-moods like "euphoric", "brooding"), `instrumentalness` (0.0-1.0), and `liveness` (0.0-1.0). Each has math-based scoring rules in the algorithm, for example, mood tags use partial overlap matching (if 1 of 2 user tags matches, you get 50% of the tag bonus).

### Challenge 2: Multiple Scoring Modes (Strategy Pattern)
Scoring weights are encapsulated in `ScoringStrategy` classes (`BalancedStrategy`, `GenreFirstStrategy`, `MoodFirstStrategy`, `EnergyFocusedStrategy`, `DiscoveryStrategy`). Each strategy defines its own weight dictionary. Switching modes in `main.py` is a single parameter change (`mode="discovery"`), keeping the code modular and extensible.

### Challenge 3: Diversity and Fairness Logic
`recommend_songs()` accepts three diversity parameters:
- `diversity_penalty` — soft penalty: deducts points for each repeat of an artist/genre already in results
- `max_per_artist` — hard cap: skips songs if the artist already has N songs in results
- `max_per_genre` — hard cap: skips songs if the genre already has N songs in results

This uses greedy re-ranking: songs are sorted by raw score, then selected one at a time with diversity constraints applied.

### Challenge 4: Visual Summary Table
Output uses the `tabulate` library with `grid` format. Each recommendation row shows rank, title, artist, genre, popularity, score, and a multi-line reasons breakdown. Section headers clearly label each experiment.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   cd src
   python main.py
   ```

### Running Tests

```bash
pytest
```

---

## Experiments You Tried

### Experiment 1: Genre-First Mode
- Doubled genre weight from 2.0 to 4.0, halved mood and energy weights.
- **Result**: Genre match completely dominated. Songs from the matching genre always took the top slots regardless of mood or energy fit. A pop song with energy 0.93 ("Gym Hero") beat a non-pop song with perfect energy match. This confirms genre weight is the most influential lever.

### Experiment 2: Energy-Focused Mode
- Tripled energy weight to 3.0, reduced genre to 0.5.
- **Result**: Songs cluster by energy level rather than genre. For the "Chill Lofi Listener," the top results were still lofi (because they naturally have low energy), but for the "High-Energy Pop Fan," high-energy songs from any genre (EDM, latin) started appearing in the top 5. Energy became the great equalizer across genres.

### Experiment 3: Mood-First Mode
- Tripled mood weight to 3.0, reduced genre to 0.5.
- **Result**: For the "Conflicting Prefs" profile (pop genre + melancholy mood + high energy 0.9), "Lonely Highway" (country, melancholy, energy 0.45) jumped to #1 despite terrible energy match, mood dominated everything. This shows how conflicting preferences get resolved differently depending on weights.

### Key Takeaway
No single weight configuration is "correct." Each mode serves different use cases: genre-first for users who know what they want, energy-focused for activity-based playlists, mood-first for emotional states.

---

## Limitations and Risks

- **Tiny catalog**: With only 20 songs, the system can't provide truly diverse recommendations. Real platforms have millions of songs.
- **Binary categorical matching**: "indie pop" and "pop" score as a complete mismatch. Real systems use embeddings that capture genre similarity.
- **No user history**: The system uses static preferences, not listening behavior. It can't learn or adapt.
- **No lyrics or language**: It ignores what songs are about, what language they're in, or cultural context.
- **Genre imbalance**: If 60% of the catalog is one genre, the system will over-recommend it to matching users while under-serving users of rarer genres.
- **Diversity penalty is optional**: Without explicitly enabling it, the system may recommend 5 songs from the same artist because it only optimizes for score, not variety.

---

## Reflection

Building this recommender taught me that even a simple scoring algorithm can produce surprisingly intuitive results, when a "Chill Lofi Listener" gets Library Rain and Midnight Coding as top picks, it genuinely feels right. But it also showed how fragile these systems are: change one weight and the entire recommendation set shifts.

The biggest insight was about bias. My system's genre weight (+2.0) means a mediocre pop song will always beat an excellent rock song for a "pop fan," even if that rock song matches every other preference perfectly. This is a microcosm of how real recommender systems create filter bubbles, they reinforce existing preferences rather than helping users discover music across boundaries. A truly fair system would need diversity constraints, similarity gradients between related genres, and a way to balance exploitation (giving users what they like) with exploration (introducing them to new things).

---

## Model Card

See [model_card.md](model_card.md) for the full model card.
