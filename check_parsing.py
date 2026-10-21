from parsing import parse_list, parse_round

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))


TEN_WORDS = "apple, dog, river, mountain, chair, cloud, bicycle, lantern, ocean, guitar"
SIX_WORDS = "apple, dog, river, mountain, chair, cloud"

# ---------------------------------------------------------------------------
# parse_list: plain comma-separated lists (start/final replies)
# ---------------------------------------------------------------------------

r = parse_list(TEN_WORDS)
check("list: normal reply -> ok, 10 words",
      r.status == "ok" and r.words == TEN_WORDS.split(", "),
      f"got {r}")

bold = "**apple**, **dog**, **river**, **mountain**, **chair**, **cloud**, **bicycle**, **lantern**, **ocean**, **guitar**"
r = parse_list(bold)
check("list: bold markers -> ok, cleaned words",
      r.status == "ok" and r.words == TEN_WORDS.split(", "),
      f"got {r}")

numbered = (
    "1. apple\n2. dog\n3. river\n4. mountain\n5. chair\n"
    "6. cloud\n7. bicycle\n8. lantern\n9. ocean\n10. guitar."
)
r = parse_list(numbered)
check("list: numbered list -> ok, numbering and trailing period stripped",
      r.status == "ok" and r.words == TEN_WORDS.split(", "),
      f"got {r}")

r = parse_list("")
check("list: empty reply -> status empty", r.status == "empty" and r.words == [],
      f"got {r}")

r = parse_list(SIX_WORDS)
check("list: only 6 words -> status too_few_words",
      r.status == "too_few_words" and r.words == SIX_WORDS.split(", "),
      f"got {r}")

# ---------------------------------------------------------------------------
# parse_round: COMMENT: ... WORDS: ... (round replies)
# ---------------------------------------------------------------------------

normal_round = f"COMMENT: these are pretty different. WORDS: {TEN_WORDS}"
r = parse_round(normal_round)
check("round: normal reply -> ok, comment + 10 words",
      r.status == "ok" and r.words == TEN_WORDS.split(", ")
      and r.comment == "these are pretty different.",
      f"got {r}")

bold_round = f"**COMMENT:** these are pretty different. **WORDS:** {bold}"
r = parse_round(bold_round)
check("round: bold markers -> ok, cleaned comment + words",
      r.status == "ok" and r.words == TEN_WORDS.split(", ")
      and r.comment == "these are pretty different.",
      f"got {r}")

numbered_round = f"COMMENT: trying to spread out.\nWORDS:\n{numbered}"
r = parse_round(numbered_round)
check("round: numbered list -> ok, numbering stripped",
      r.status == "ok" and r.words == TEN_WORDS.split(", ")
      and r.comment == "trying to spread out.",
      f"got {r}")

r = parse_round("")
check("round: empty reply -> status empty", r.status == "empty" and r.words == [],
      f"got {r}")

six_round = f"COMMENT: ran out of ideas. WORDS: {SIX_WORDS}"
r = parse_round(six_round)
check("round: only 6 words -> status too_few_words",
      r.status == "too_few_words" and r.words == SIX_WORDS.split(", "),
      f"got {r}")

r = parse_round("I think these words work well together.")
check("round: no WORDS: label -> status unreadable",
      r.status == "unreadable" and r.words == [],
      f"got {r}")


print(f"{'Check':<55} {'Status':<6} Detail")
print("-" * 110)
for name, status, detail in results:
    print(f"{name:<55} {status:<6} {detail}")
