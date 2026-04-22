# Model Card: Music Recommender Agent

## Model / System Overview

This system combines a **deterministic content-based recommender** (Modules 1-3) with an **LLM-driven agentic workflow** (Module 4). The deterministic core uses weighted feature scoring across 11 song attributes (genre, mood, energy, valence, danceability, acousticness, popularity, decade, mood tags, instrumentalness, liveness) with five selectable scoring strategies. The Module 4 agent uses Claude Haiku 4.5 (via the Anthropic API) to parse free-form natural language requests, retrieve context from a curated knowledge base, select the appropriate scoring strategy and preference profile, and generate a conversational explanation of the results.

**Intended use:** Educational demonstration of AI-augmented recommendation systems. The system is designed for single-user, local use and is not production-grade.

**Out of scope:** Multi-user deployment, real-time audio analysis, collaborative filtering, user history tracking, or any use with real user data.

---

## Limitations and Biases

### Inherent limitations

1. **Tiny catalog.** The system only has 20 songs. No recommendation engine can provide diverse results from a pool this small. A user who asks for "classical music" or "hip hop" will get irrelevant results because those genres don't exist in the catalog.

2. **Binary genre matching.** "Indie pop" and "pop" are treated as a complete mismatch (+0.0 vs +2.0). Real systems use learned genre embeddings that capture that "indie pop" is much closer to "pop" than it is to "metal." This bias systematically underscores songs from similar-but-not-identical genres.

3. **English-only, Western-centric catalog.** Every song in the catalog is described with English-language mood tags, Western genre labels, and Western music-theory features (valence, danceability). The system has no concept of non-Western musical traditions.

4. **LLM hallucination risk.** Although the architecture is designed so the LLM never invents scores or song titles (it must call the `recommend_songs` tool), the LLM *does* construct the user preference profile from the query. If the LLM misinterprets "I'm feeling blue" as `mood="happy"` (unlikely but possible), the deterministic core will dutifully return wrong results. The self-check/confidence evaluator partially mitigates this by verifying the genre was satisfied, but it can't detect all semantic misparses.

5. **Knowledge base freshness.** The RAG corpus is hand-written and static. If the `songs.csv` were updated (new artists, new genres), the knowledge base would need manual updating. There's no automatic sync.

6. **Heuristic fallback is simplistic.** The keyword-based fallback parser (`_heuristic_prefs_from_query`) maps "workout" to `energy=0.9` and "lofi" to `genre=lofi`, but can't handle nuanced requests like "something between jazz and lofi." The fallback exists as a safety net, not a replacement for the LLM.

### Potential biases

- **Genre weight dominance (+2.0 in balanced mode):** A mediocre pop song always beats an excellent rock song for a "pop fan." This reinforces filter bubbles.
- **Popularity bias in the catalog:** Average popularity in the CSV is 63. The system defaults to recommending more popular songs unless `discovery` mode is explicitly selected.
- **Mood tag overlap is limited to exact string match:** "Happy" and "joyful" are treated as unrelated. The retriever helps somewhat (the LLM reads the moods.md doc and learns that "euphoric" is adjacent to "uplifting"), but the deterministic scoring still does exact-match.

---

## Misuse Potential and Mitigations

### Could this system be misused?

In its current form, the risk is low, it's a local CLI tool with 20 songs. But scaled up, a system like this could:

1. **Amplify filter bubbles** by only recommending what users already like, reducing musical diversity.
2. **Manipulate listening behavior** if the scoring weights were tuned to favor songs from paying labels or artists.
3. **Leak user preferences** if query logs were stored and shared without consent.

### Mitigations built into this system

- **Diversity penalty system** (Module 1-3) is available and documented. Users can explicitly enable it to break out of repetitive recommendations.
- **Discovery mode** penalizes popular tracks, providing an explicit mechanism to surface underrepresented catalog entries.
- **Full logging and traceability** — every tool call, every retrieved document, every confidence score is logged to `logs/agent.log`. There is no hidden decision-making.
- **Deterministic core** — the LLM never invents scores. Any recommendation can be independently verified by hand using the CSV and the scoring formula in `recommender.py`.

---

## Testing Results

| Metric | Value | Notes |
|--------|-------|-------|
| Unit tests | 30/30 passed | 13 Module 1-3 + 7 retriever + 10 agent/evaluator/tool |
| Eval harness (fallback) | 0/6 "full pass" | Expected: every case fails `retrieval_used` check |
| Eval harness (non-RAG checks) | 8/10 passed | Genre match and energy-range assertions |
| Avg confidence (fallback) | 0.49 | Expected to exceed 0.75 with live API |
| Fallback recovery | Verified | Real API billing error was caught and recovered |

The eval harness is designed so that with a valid API key and Claude Haiku 4.5 credits, all 6 cases should pass (the LLM selects correct strategies + retrieval is used).

---

## What Surprised Me While Testing Reliability

Three things stood out:

1. **The fallback was more useful than I expected.** I built the heuristic fallback (`_heuristic_prefs_from_query`) as a safety net, assuming I'd only need it for CI environments without API keys. Then the Anthropic API threw a real billing error mid-development. The fallback kicked in silently and still returned 5 lofi songs for a "chill lofi for studying" query — with confidence 0.57, which the evaluator correctly marked as `ok=True`. I hadn't planned to stress-test the fallback with a real API failure, but it happened naturally, and the system passed. That convinced me that graceful degradation is not optional in AI systems.

2. **Confidence scores exposed the heuristic's blind spots.** In the eval harness, the "hidden gems" prompt (`"surface some hidden gems, nothing on the charts"`) got confidence 0.47 in fallback mode. The evaluator correctly flagged it: the heuristic parser doesn't know the word "hidden" means "low popularity" or that it should select `discovery` mode. Confidence scoring turned a subjective "the fallback isn't great at nuance" into a measurable 0.47 that I could compare across runs. This was surprising because I initially thought of confidence as a user-facing display value, not a diagnostic signal.

3. **The retriever's ranking was fragile without stopwords.** Early on, a query like "chill lofi for studying late at night" was returning the "country" genre doc as a top hit because "for" and "a" appeared frequently in every document. Adding a 30-word stopword list fixed the ranking entirely. This surprised me because TF-IDF is a textbook technique — I assumed it would "just work." It reminded me that even simple NLP components need tuning for the specific corpus.

---

## AI Collaboration Reflection

### One instance where AI was helpful

When designing the retriever, I initially planned to use scikit-learn's `TfidfVectorizer`. Claude (the AI I collaborated with during development) suggested a pure-Python implementation using bag-of-words term frequencies and IDF computed manually. This was the right call — the corpus is only ~40 paragraphs, and avoiding sklearn kept the install frictionless and the code self-contained. The resulting retriever is 80 lines and has zero external dependencies.

### One instance where AI was flawed

Early in the project, Claude suggested embedding the knowledge base passages directly into the Claude system prompt as a giant multi-paragraph string, rather than dynamically retrieving relevant passages. This would have (a) blown out the input token count on every request, (b) made the RAG component non-functional (the model would see everything, not just relevant context), and (c) defeated the purpose of the retriever entirely. I recognized the flaw and designed the pipeline so the retriever pre-selects the top 3 passages and only those get injected into the system prompt. This is what makes it real RAG rather than just "paste the docs."

---

## Ethical Reflection

AI isn't neutral infrastructure, it reflects the choices of whoever designed the scoring formula, curated the catalog, and wrote the prompts. In this system, the genre weight of +2.0 is an *editorial decision* that says "genre matters twice as much as mood." That's defensible for some users and wrong for others. The Strategy pattern lets users override it, but the *default* still shapes most users' experience. Real recommender systems face this at a much larger scale: Spotify's "Discover Weekly" shapes what millions of people listen to, and the weights are invisible to users.

The biggest lesson: **transparency is a design choice, not an afterthought.** This system logs every tool call, shows confidence scores, and explains *why* each song was recommended. Most production recommender systems don't. If I were building this for real, I'd want users to be able to see and adjust the weights, not just accept whatever the algorithm decided behind the scenes.
