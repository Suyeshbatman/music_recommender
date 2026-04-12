# Reflection: Profile Comparison Notes

## High-Energy Pop Fan vs. Chill Lofi Listener

These two profiles are near-opposites on the energy spectrum (0.85 vs 0.35) and prefer completely different genres and moods. The Pop Fan's top pick was Sunrise City (score 4.79), an upbeat, high-energy pop track. The Lofi Listener's top pick was Library Rain (score 4.73), a low-energy, acoustic lofi track. There is zero overlap in their top 5 lists. This makes sense: the combination of different genre, mood, AND energy means these users live in entirely different musical worlds. The system correctly separates them.

## Deep Intense Rock vs. EDM Party Mode

Both profiles want high energy (0.92 vs 0.95), but they diverge on genre (rock vs edm) and mood (intense vs energetic). Interestingly, Gym Hero appeared in the Rock profile's top 5 (as #3) because it's intense + high-energy, even though it's pop. For the EDM profile, Bass Cathedral ranked #1. A perfect genre + mood + energy match. The overlap song is Gym Hero, which shows that high energy alone creates some cross-genre recommendations. The system correctly differentiates their primary recommendations based on genre, but the shared "high energy" preference creates a small overlap zone.

## Mellow Jazz Lover vs. Chill Lofi Listener

These profiles are interesting because they share similar energy levels (0.40 vs 0.35) and both prefer relaxed/chill moods, but they differ on genre (jazz vs lofi) and acoustic preference (both like acoustic). The Jazz Lover gets Blue Note Cafe and Coffee Shop Stories as top picks, both jazz tracks from Slow Stereo. The Lofi Listener gets Library Rain and Midnight Coding. Despite similar vibes, genre match keeps them in separate lanes. This demonstrates that our system can distinguish between "chill jazz" and "chill lofi" users, which is good. These are genuinely different listening experiences.

## Conflicting Prefs vs. High-Energy Pop Fan

Both profiles prefer pop and high energy, but they diverge on mood (melancholy vs happy). In balanced mode, the Conflicting profile's results look almost identical to the Pop Fan's Gym Hero and Sunrise City dominate because genre weight overwhelms the mood difference. The melancholy preference barely affects results because there's only one melancholy song (Lonely Highway, a country track) and it gets buried by genre mismatch. This reveals a real weakness: the system can't find "sad pop" because the catalog doesn't have any, and it doesn't understand that "moody synthwave" might be close to "melancholy pop."

## EDM Party Mode vs. No Strong Preference

The EDM profile gets crisp, confident recommendations (Bass Cathedral at 4.89) because the catalog has two EDM tracks that match perfectly. The "No Strong Preference" profile (indie pop, nostalgic, 0.5 energy) gets mediocre scores across the board. The top pick is Rooftop Lights at only 3.47, and the remaining picks feel scattered. This comparison shows that the system works best for users with strong, well-represented preferences, and struggles when the user's taste doesn't align cleanly with the catalog's distribution.

## Balanced Mode vs. Energy-Focused Mode (same profile)

Running the High-Energy Pop Fan in both modes reveals how weight shifts change everything. In balanced mode, the top 5 includes songs from 4 different genres (pop, edm, indie pop, latin) with Sunrise City at #1. In energy-focused mode, the list reshuffles, Salsa del Sol jumps higher because its energy (0.87) is close to the target, even though it's latin. The genre "moat" disappears when you reduce its weight. This is a practical demonstration of the exploration-exploitation tradeoff: genre-first exploits known preferences, energy-focused explores across genre boundaries.
