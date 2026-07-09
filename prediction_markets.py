"""
Free prediction-market signal fetchers (no API keys required).

Primary source: Polymarket Gamma API (public-search).
Fallback: regex extraction from Tavily research text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "will", "the", "a", "an", "in", "on", "by", "to", "of", "be", "is", "are",
    "was", "were", "at", "or", "and", "for", "as", "it", "its", "this", "that",
    "than", "before", "after", "from", "with", "have", "has", "had", "not",
    "any", "all", "what", "which", "who", "how", "when", "where", "there",
})

_POLYMARKET_SEARCH = "https://gamma-api.polymarket.com/public-search"


@dataclass(frozen=True)
class MarketSignal:
    yes_probability: float
    source: str
    market_title: str
    match_score: float
    volume: float = 0.0


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _get_json(url: str, timeout_s: int = 20) -> dict | list:
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "metaculus-forecast-bot/1.0",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _parse_polymarket_yes_price(market: dict) -> float | None:
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        prices = json.loads(market.get("outcomePrices", "[]"))
    except (json.JSONDecodeError, TypeError):
        return None
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    for outcome, price in zip(outcomes, prices):
        if str(outcome).lower() == "yes":
            p = float(price)
            if 0.0 < p < 1.0:
                return p
    return None


def fetch_polymarket_signals(
    question_text: str,
    limit: int = 5,
    timeout_s: int = 20,
) -> list[MarketSignal]:
    """Search Polymarket for markets matching a Metaculus question."""
    query = question_text[:160].strip()
    if not query:
        return []

    url = f"{_POLYMARKET_SEARCH}?q={quote(query)}&limit_per_type={limit}"
    try:
        data = _get_json(url, timeout_s=timeout_s)
    except Exception as exc:
        logger.warning(f"Polymarket search failed: {type(exc).__name__}")
        return []

    if not isinstance(data, dict):
        return []

    q_tokens = _tokenize(question_text)
    signals: list[MarketSignal] = []

    for event in data.get("events", []) or []:
        title = (event.get("title") or "").strip()
        if not title:
            continue
        score = _jaccard(q_tokens, _tokenize(title))
        for market in event.get("markets", []) or []:
            yes_p = _parse_polymarket_yes_price(market)
            if yes_p is None:
                continue
            vol = float(market.get("volumeNum") or market.get("volume") or 0)
            signals.append(MarketSignal(
                yes_probability=yes_p,
                source="polymarket",
                market_title=title,
                match_score=score,
                volume=vol,
            ))

    signals.sort(key=lambda s: s.match_score * (1.0 + min(s.volume, 1_000_000) ** 0.1),
                 reverse=True)
    return signals[:limit]


def parse_probabilities_from_text(text: str) -> list[float]:
    """Extract percentage probabilities from research snippets (Tavily fallback)."""
    found: list[float] = []
    for match in re.finditer(
        r"(?:probability|odds|chance|market|polymarket|kalshi|metaforecast)"
        r"[^.\n]{0,60}?(\d{1,2}(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    ):
        val = float(match.group(1)) / 100.0
        if 0.01 <= val <= 0.99:
            found.append(val)
    return found


def aggregate_market_signal(
    polymarket_signals: list[MarketSignal],
    research_text: str = "",
    min_match_score: float = 0.20,
) -> MarketSignal | None:
    """
    Return the best prediction-market signal, or None if nothing is reliable.
    """
    if polymarket_signals:
        best = polymarket_signals[0]
        if best.match_score >= min_match_score:
            return best

    parsed = parse_probabilities_from_text(research_text)
    if parsed:
        median = float(sorted(parsed)[len(parsed) // 2])
        return MarketSignal(
            yes_probability=median,
            source="research_text",
            market_title="(parsed from Tavily research)",
            match_score=0.15,
        )
    return None


def blend_with_market(
    model_probability: float,
    market: MarketSignal | None,
    weight: float,
) -> float:
    """Linear blend of model forecast with a prediction-market signal."""
    if market is None or weight <= 0:
        return model_probability
    w = min(1.0, max(0.0, weight))
    # Weight market influence by match confidence
    effective_w = w * min(1.0, market.match_score / 0.35)
    return (1.0 - effective_w) * model_probability + effective_w * market.yes_probability


def format_market_context(signals: list[MarketSignal], aggregate: MarketSignal | None) -> str:
    """Human-readable block for research prompts."""
    if not signals and aggregate is None:
        return ""
    lines = ["--- Prediction market signals ---"]
    for sig in signals[:3]:
        lines.append(
            f"- [{sig.source}] {sig.market_title}: "
            f"YES {sig.yes_probability:.1%} "
            f"(match={sig.match_score:.2f}, vol={sig.volume:,.0f})"
        )
    if aggregate:
        lines.append(
            f"Best signal for blending: {aggregate.yes_probability:.1%} "
            f"from {aggregate.source}"
        )
    return "\n".join(lines)
