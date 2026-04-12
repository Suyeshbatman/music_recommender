# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 2.0** (with Optional Extensions)

---

## 2. Intended Use

This system suggests 3–5 songs from a small catalog of 20 songs based on a user's preferred genre, mood, energy level, acoustic taste, and advanced features (popularity, release decade, detailed mood tags, instrumentalness, liveness). It supports 5 scoring strategies via a Strategy pattern, and includes a diversity penalty system to prevent repetitive results. It is designed for classroom exploration and learning about how recommender systems work. It is not intended for real production use, real user personalization, or commercial music distribution.

**Not intended for:**
- Replacing real music recommendation engines
- Making decisions about music licensing, royalties, or artist promotion
- Profiling real users or inferring personal attributes from listening habits

---

## 3. How the Model Works

The system reads a catalog of songs from a CSV file. Each song has attributes like genre, mood, energy (0–1 scale), valence, danceability, and acousticness.

When a user provides their preferences (favorite genre, mood, target energy, and whether they like acoustic music), the system scores every song in the catalog against those preferences. The scoring works like a points system:

- A song gets the most points for matching the user's favorite genre (like getting bonus points on a quiz for the right category).
- It gets additional points for matching the user's preferred mood.
- It gets points based on how close the song's energy level is to what the user wants. A song at 0.8 energy for a user who wants 0.85 scores almost perfectly, while a song at 0.2 barely scores at all.
- Smaller bonuses come from acoustic fit, valence (how positive the song sounds), and danceability.

After every song has a score, the system sorts them from highest to lowest and shows the top results, along with a breakdown of why each song earned its score.

The system also supports five "modes" via a Strategy pattern (balanced, genre-first, mood-first, energy-focused, discovery) that change how much each feature matters. The "discovery" mode actually penalizes popular songs, surfacing hidden gems. Additionally, the system has an optional diversity penalty: when enabled, it deducts points from songs if their artist or genre already appears in the results, or enforces hard caps like "max 1 song per artist." This prevents the top 5 from being dominated by a single artist or genre.

---

## 4. Data

- **Catalog size**: 20 songs
- **Source**: Manually created CSV file (`data/songs.csv`)
- **Genres represented**: pop, lofi, rock, ambient, jazz, synthwave, indie pop, edm, country, metal, r&b, folk, latin
- **Moods represented**: happy, chill, intense, relaxed, focused, moody, energetic, melancholy, romantic, nostalgic
- **Features per song**: 15 (id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_decade, mood_tags, instrumentalness, liveness)
- **Expanded from**: The starter file had 10 songs with 10 features. Expanded to 20 songs with 15 features including advanced attributes (Challenge 1).
- **Missing from the data**: Hip-hop, classical, K-pop, and many global genres are absent. The dataset reflects a narrow slice of musical taste. There are no songs in languages other than English (implied). The "mood" labels and "mood_tags" are subjective and assigned by one person.

---

## 5. Strengths

- **Transparency**: Every recommendation comes with a full breakdown of why it was chosen. Users can see exactly which features contributed how many points. This is more explainable than a neural network.
- **Intuitive results for clear preferences**: When a user has a strong, consistent profile (e.g., "lofi + chill + low energy"), the top results feel correct — Library Rain and Midnight Coding are genuinely chill lofi tracks.
- **Configurable**: Five scoring strategies via Strategy pattern let users see how weight changes affect outcomes.
- **Diversity-aware**: The optional diversity penalty prevents the same artist or genre from dominating results, a real-world concern addressed in Challenge 3.
- **Rich features**: Mood tags enable partial-match scoring (e.g., matching 1 of 2 tags gives 50% bonus), which is more nuanced than binary mood matching alone.
- **Simplicity**: The algorithm is easy to understand, debug, and modify. There are no black boxes.

---

## 6. Limitations and Bias

- **Genre dominance**: With a +2.0 weight, genre match is the single strongest signal. A mediocre same-genre song will almost always beat a perfect-fit song from a different genre. This creates a filter bubble.
- **Binary categorical matching**: "indie pop" and "pop" are treated as completely unrelated. Real systems understand that these genres are close cousins. This means users who like "indie pop" get no credit for pop songs that might suit them.
- **Small catalog bias**: With only 20 songs, some genres have only 1–2 representatives. A "country" fan only has one song to choose from, making the system trivially bad for underrepresented genres.
- **No temporal or contextual awareness**: The system doesn't know if it's morning or night, if the user is working out or studying, or what they listened to recently.
- **Static preferences**: Real users' tastes shift by mood, time, and social context. This system assumes preferences are fixed.
- **Subjective mood labels**: One person's "chill" is another person's "boring." The mood tags are not validated against listener perception.

---

## 7. Evaluation

### Profiles Tested

| Profile | Genre | Mood | Energy | Acoustic |
|---------|-------|------|--------|----------|
| High-Energy Pop Fan | pop | happy | 0.85 | No |
| Chill Lofi Listener | lofi | chill | 0.35 | Yes |
| Deep Intense Rock | rock | intense | 0.92 | No |
| Mellow Jazz Lover | jazz | relaxed | 0.40 | Yes |
| EDM Party Mode | edm | energetic | 0.95 | No |
| Conflicting Prefs | pop | melancholy | 0.90 | No |
| No Strong Preference | indie pop | nostalgic | 0.50 | No |

### What Surprised Me

- The "Conflicting Prefs" profile (pop + melancholy + high energy) was the most revealing. In balanced mode, genre dominated and pop songs ranked first despite not matching the melancholy mood. In mood-first mode, "Lonely Highway" (country, melancholy) jumped to #1 despite terrible energy fit. This shows that conflicting preferences expose the system's inability to blend multiple signals gracefully.
- The "EDM Party Mode" profile correctly placed Bass Cathedral at #1 (4.89 points), the system nailed this one. The genre + mood + energy all aligned, and it felt like a genuine recommendation.
- The "No Strong Preference" profile produced the least satisfying results. With only one indie pop song in the catalog, the system had nothing to fall back on. The remaining recommendations felt random.

### Experiments

Five scoring strategies were tested (balanced, genre-first, energy-focused, mood-first, discovery). Each produced meaningfully different rankings, confirming that the system is sensitive to weight configuration. No single configuration was universally "best." The discovery mode was particularly interesting, it penalized popular tracks, causing hidden gems like "Autumn Letters" (popularity 38) to surface above mainstream picks.

### Diversity Experiments (Challenge 3)

Tested the Mellow Jazz Lover profile three ways:
- **No diversity**: Both jazz songs from Slow Stereo appeared at #1 and #2
- **Soft penalty (1.0)**: The second Slow Stereo song was penalized, pushing other artists up
- **Hard cap (max 1 per artist)**: Forced 5 unique artists in the top 5, surfacing songs from Paper Lanterns, LoRoom, and Orbit Bloom that were previously hidden

---

## 8. Future Work

1. **Genre similarity scores**: Instead of binary match/no-match, use a similarity matrix where "indie pop" and "pop" score 0.7 similarity, "rock" and "metal" score 0.8, etc. This would reduce the filter bubble effect.
2. **User history simulation**: Track which songs a simulated user has already heard and boost novelty (songs they haven't encountered).
3. **Larger dataset**: Expand to 100+ songs with better genre balance so underrepresented genres get fair coverage.
4. **Collaborative filtering layer**: Simulate multiple users and use "users like you also enjoyed" as an additional signal.
5. **Web UI with sliders**: Build a Streamlit or Flask interface where users can adjust weights with sliders and see recommendations update in real time.

---

## 9. Personal Reflection

The biggest learning moment was seeing how a 6-line scoring function can produce results that "feel" like real recommendations. When I ran the Chill Lofi profile and got Library Rain and Midnight Coding at the top, it genuinely felt like something Spotify would suggest. That was surprising, I expected such a simple algorithm to produce obviously bad results, but the combination of genre match + mood match + energy similarity covers a lot of ground.

Using AI tools helped me rapidly prototype the scoring logic and generate diverse test songs for the CSV. But I had to double-check the math, the AI initially suggested absolute energy values rather than energy similarity (distance from target), which would have rewarded high-energy songs for everyone. The correction to use `1.0 - abs(difference)` was critical.

What surprised me most is that the system's failures are more interesting than its successes. The "Conflicting Prefs" profile exposed a real design tension: when a user's preferences contradict each other, there's no objectively correct answer, it depends on which dimension you prioritize. Real recommender systems face this constantly, and the answer is often "show a diverse mix," which my system can't do without a diversity constraint.

Implementing the optional extensions (advanced features, Strategy pattern, diversity penalty, tabulate output) deepened my understanding significantly. The Strategy pattern was eye-opening. It showed how the same algorithm with different weight configurations is really a family of recommenders, not just one. The diversity penalty was the most satisfying to build because it solved a real problem I noticed during testing (two Slow Stereo jazz songs always hogging the top spots for jazz lovers). And the discovery mode, penalizing popularity, was a fun inversion that made me think about how real platforms could help surface independent artists instead of always promoting the most-streamed tracks.
