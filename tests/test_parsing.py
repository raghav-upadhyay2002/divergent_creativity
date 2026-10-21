from dat.parsing import parse_list, parse_round

TEN_WORDS = "apple, dog, river, mountain, chair, cloud, bicycle, lantern, ocean, guitar"
SIX_WORDS = "apple, dog, river, mountain, chair, cloud"
BOLD_TEN_WORDS = ("**apple**, **dog**, **river**, **mountain**, **chair**, **cloud**, "
                  "**bicycle**, **lantern**, **ocean**, **guitar**")
NUMBERED_TEN_WORDS = (
    "1. apple\n2. dog\n3. river\n4. mountain\n5. chair\n"
    "6. cloud\n7. bicycle\n8. lantern\n9. ocean\n10. guitar."
)


# ---------------------------------------------------------------------------
# parse_list: plain comma-separated lists (start/final replies)
# ---------------------------------------------------------------------------

def test_list_normal_reply():
    """A plain comma-separated list of 10 words parses as ok."""
    r = parse_list(TEN_WORDS)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")


def test_list_bold_markers_are_stripped():
    """Bold markers around each word are cleaned off."""
    r = parse_list(BOLD_TEN_WORDS)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")


def test_list_numbered_list():
    """Numbering and a trailing period are stripped from each word."""
    r = parse_list(NUMBERED_TEN_WORDS)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")


def test_list_empty_reply():
    """An empty reply parses with status empty."""
    r = parse_list("")
    assert r.status == "empty"
    assert r.words == []


def test_list_too_few_words():
    """Only 6 words parses with status too_few_words."""
    r = parse_list(SIX_WORDS)
    assert r.status == "too_few_words"
    assert r.words == SIX_WORDS.split(", ")


# ---------------------------------------------------------------------------
# parse_round: COMMENT: ... WORDS: ... (round replies)
# ---------------------------------------------------------------------------

def test_round_normal_reply():
    """A normal COMMENT:/WORDS: reply parses the comment and 10 words."""
    reply = f"COMMENT: these are pretty different. WORDS: {TEN_WORDS}"
    r = parse_round(reply)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")
    assert r.comment == "these are pretty different."


def test_round_bold_markers_are_stripped():
    """Bold markers around COMMENT:/WORDS: and the words are cleaned off."""
    reply = f"**COMMENT:** these are pretty different. **WORDS:** {BOLD_TEN_WORDS}"
    r = parse_round(reply)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")
    assert r.comment == "these are pretty different."


def test_round_numbered_list():
    """Numbering in the WORDS: section is stripped."""
    reply = f"COMMENT: trying to spread out.\nWORDS:\n{NUMBERED_TEN_WORDS}"
    r = parse_round(reply)
    assert r.status == "ok"
    assert r.words == TEN_WORDS.split(", ")
    assert r.comment == "trying to spread out."


def test_round_empty_reply():
    """An empty reply parses with status empty."""
    r = parse_round("")
    assert r.status == "empty"
    assert r.words == []


def test_round_too_few_words():
    """Only 6 words in WORDS: parses with status too_few_words."""
    reply = f"COMMENT: ran out of ideas. WORDS: {SIX_WORDS}"
    r = parse_round(reply)
    assert r.status == "too_few_words"
    assert r.words == SIX_WORDS.split(", ")


def test_round_no_words_label():
    """A reply with no WORDS: label parses with status unreadable."""
    r = parse_round("I think these words work well together.")
    assert r.status == "unreadable"
    assert r.words == []
