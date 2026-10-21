import pandas as pd

from dat.scoring import dat_score, score_details, valid_words


def test_two_words_min_words_2():
    """cat, dog with min_words=2 scores about 19.83."""
    score = dat_score("cat, dog", min_words=2)
    assert score is not None
    assert abs(score - 19.83) < 0.01


def test_body_parts():
    """Seven body-part words score about 50.31."""
    score = dat_score("arm, eyes, feet, hand, head, leg, body")
    assert score is not None
    assert abs(score - 50.31) < 0.1


def test_varied_words_high_score():
    """Seven unrelated words score about 86.67."""
    score = dat_score("lantern, justice, sandwich, volcano, algorithm, umbrella, nostalgia")
    assert score is not None
    assert abs(score - 86.67) < 0.1


def test_varied_words_very_high_score():
    """Another set of seven unrelated words scores about 95.22."""
    score = dat_score("hippo, jumper, machinery, prickle, tickets, tomato, violin")
    assert score is not None
    assert abs(score - 95.22) < 0.1


def test_too_few_real_words_returns_none():
    """Only 3 real words among 7 means no score."""
    score = dat_score("cat, dog, tree, xqzt, blorp, zzyx, qwrpt")
    assert score is None


def test_repeats_collapse_to_one_word():
    """"Apple", "apple!" and " apple " all clean to one "apple", not three."""
    words = ["Apple", "apple!", " apple ", "dog", "cat", "bird", "fish", "lion", "wolf"]
    used = valid_words(words)
    assert used.count("apple") == 1
    assert len(used) == 7


def test_matches_original_dat_column():
    """Re-scoring 100 random historical rows should match their saved dat column."""
    df = pd.read_csv("a9v2t-osfstorage-archive/data/study_1a_data.csv")
    sample = df.sample(100, random_state=0)
    noun_cols = [f"noun_{n}" for n in range(10)]

    max_diff = 0.0
    for _, row in sample.iterrows():
        words = [row[c] for c in noun_cols if pd.notna(row[c])]
        computed = dat_score(words)
        expected = row["dat"]

        if computed is None or pd.isna(expected):
            assert computed is None and pd.isna(expected)
            continue

        diff = abs(float(computed) - float(expected))
        max_diff = max(max_diff, diff)

    # float32 rounding noise, not a logic mismatch -- see dat/scoring.py's
    # _mean_pairwise_distance_score for why this isn't bit-exact
    assert max_diff < 0.001


def test_score_all10_requires_at_least_7_valid_words():
    """score_all10 needs 7+ valid words, same floor as the main score."""
    from dat.scoring import score_details_all10

    score, used, _dropped = score_details_all10("cat, dog, tree, xqzt, blorp, zzyx, qwrpt")
    assert score is None
    assert len(used) < 7


def test_score_all10_uses_up_to_10_words():
    """score_all10 can use up to 10 valid words, not just the first 7."""
    from dat.scoring import score_details_all10

    words = "apple, dog, river, mountain, chair, cloud, bicycle, lantern, ocean, guitar"
    score, used, _dropped = score_details_all10(words)
    assert score is not None
    assert len(used) == 10
