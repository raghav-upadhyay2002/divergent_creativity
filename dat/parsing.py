import re
from dataclasses import dataclass, field
from typing import List, Optional

MIN_WORDS = 10


@dataclass
class ParseResult:
    status: str  # "ok", "empty", "too_few_words", "unreadable"
    words: List[str] = field(default_factory=list)
    comment: Optional[str] = None


def _clean_token(token):
    """Strip whitespace, numbering, bullets and trailing periods from one word."""
    token = token.strip()
    token = re.sub(r"^\d+[\.\)]\s*", "", token)  # leading numbering: "1. " or "1) "
    token = re.sub(r"^[-•]\s*", "", token)  # leading bullet: "- " or "• "
    token = token.strip().strip(".").strip()
    return token


def _split_words(text):
    """Split text on commas/newlines and clean each resulting word."""
    raw_tokens = re.split(r"[,\n]+", text)
    words = [_clean_token(t) for t in raw_tokens]
    return [w for w in words if w]


def parse_list(reply):
    """Parse a plain comma-separated list of words (start/final replies)."""
    if reply is None or not reply.strip():
        return ParseResult(status="empty")

    text = re.sub(r"\*+", "", reply)  # strip bold markers like **WORDS:**
    words = _split_words(text)

    if len(words) < MIN_WORDS:
        return ParseResult(status="too_few_words", words=words)

    return ParseResult(status="ok", words=words)


def parse_round(reply):
    """Parse a round reply with a COMMENT: section and a WORDS: section."""
    if reply is None or not reply.strip():
        return ParseResult(status="empty")

    text = re.sub(r"\*+", "", reply)  # strip bold markers like **COMMENT:** / **WORDS:**

    words_match = re.search(r"WORDS\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not words_match:
        return ParseResult(status="unreadable")

    comment_match = re.search(
        r"COMMENT\s*:\s*(.*?)(?=WORDS\s*:|$)", text, re.IGNORECASE | re.DOTALL)
    comment = comment_match.group(1).strip() if comment_match else None

    words = _split_words(words_match.group(1))

    if len(words) < MIN_WORDS:
        return ParseResult(status="too_few_words", words=words, comment=comment)

    return ParseResult(status="ok", words=words, comment=comment)
