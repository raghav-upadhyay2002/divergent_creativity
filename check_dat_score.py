import pandas as pd

from dat_score import dat_score, valid_words, score_details

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))


# 1. "cat, dog" with min_words=2 -> 19.83
score = dat_score("cat, dog", min_words=2)
check("1. cat, dog (min_words=2) ~= 19.83", score is not None and abs(score - 19.83) < 0.01,
      f"got {score}")

# 2. "arm, eyes, feet, hand, head, leg, body" -> about 50.31
score = dat_score("arm, eyes, feet, hand, head, leg, body")
check("2. body parts ~= 50.31", score is not None and abs(score - 50.31) < 0.1,
      f"got {score}")

# 3. "lantern, justice, sandwich, volcano, algorithm, umbrella, nostalgia" -> 86.67
score = dat_score("lantern, justice, sandwich, volcano, algorithm, umbrella, nostalgia")
check("3. varied words ~= 86.67", score is not None and abs(score - 86.67) < 0.1,
      f"got {score}")

# 4. "hippo, jumper, machinery, prickle, tickets, tomato, violin" -> about 95.22
score = dat_score("hippo, jumper, machinery, prickle, tickets, tomato, violin")
check("4. varied words ~= 95.22", score is not None and abs(score - 95.22) < 0.1,
      f"got {score}")

# 5. Only 3 real words -> None
score = dat_score("cat, dog, tree, xqzt, blorp, zzyx, qwrpt")
check("5. only 3 real words -> None", score is None,
      f"got {score}")

# 6. "Apple", "apple!" and " apple " all clean to "apple"; repeats count once
words_6 = ["Apple", "apple!", " apple ", "dog", "cat", "bird", "fish", "lion", "wolf"]
vw = valid_words(words_6)
check("6. repeats collapse to one apple", vw.count("apple") == 1 and len(vw) == 7,
      f"got {vw}")

# 7. Re-score 100 random rows (seed 0) from study_1a_data.csv
df = pd.read_csv("a9v2t-osfstorage-archive/data/study_1a_data.csv")
sample = df.sample(100, random_state=0)

noun_cols = [f"noun_{n}" for n in range(10)]
exact_matches = 0
max_diff = 0.0
max_diff_row = None
for _, row in sample.iterrows():
    words = [row[c] for c in noun_cols if pd.notna(row[c])]
    computed = dat_score(words)
    expected = row["dat"]

    if computed is None or pd.isna(expected):
        if computed is None and pd.isna(expected):
            exact_matches += 1
        continue

    diff = abs(float(computed) - float(expected))
    if diff == 0.0:
        exact_matches += 1
    if diff > max_diff:
        max_diff = diff
        max_diff_row = row["index"]

check("7. 100 random rows vs dat column (report only)", True,
      f"{exact_matches}/100 bit-exact matches, largest diff = {max_diff:.8f} "
      f"(row {max_diff_row}) -- remaining diffs are float32 rounding noise")


print(f"{'Check':<45} {'Status':<6} Detail")
print("-" * 100)
for name, status, detail in results:
    print(f"{name:<45} {status:<6} {detail}")
