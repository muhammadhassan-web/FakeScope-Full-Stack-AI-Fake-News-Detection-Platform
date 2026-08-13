"""Live fact-checking — web search + LLM reasoning over retrieved evidence.

Unlike the offline TF-IDF classifier in predictor.py, this hits the live web
so it can speak to claims about current events. It never reasons from the
model's own "knowledge" — only from sources fetched at request time, and the
verdict is explicitly grounded in citations back to those sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
SEARCH_TIMEOUT = 15
LLM_TIMEOUT = 30
MAX_SNIPPET_CHARS = 600

VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "UNVERIFIED"}

SYSTEM_PROMPT = """You are a careful fact-checking assistant. You are given TODAY'S DATE, a \
CLAIM, and a numbered list of SOURCES retrieved from a live web search just now. Decide \
whether the CLAIM is supported, contradicted, or cannot be verified — using ONLY the \
provided sources, never outside knowledge.

Rules:
- Each source lists its publish date (or "unknown" if not available). If the claim has a \
time component (e.g. "today", "this week", "currently"), the source's date must actually \
match that window relative to TODAY'S DATE — a real event from a different day does NOT \
support a claim about "today". Sources with an unknown or mismatched date cannot support a \
time-specific claim; treat that as UNVERIFIED (or note the mismatch if it actively \
contradicts the claim's timing).
- If the sources don't clearly address the claim, or conflict with each other, respond UNVERIFIED.
- Only cite source numbers that were actually given to you.
- Keep the explanation to 2-3 sentences, reference sources like [1], [2], and if a date \
mismatch is why you're not confirming, say so explicitly.
- Respond with ONLY a JSON object, no other text, matching exactly:
{"verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED", "confidence": <integer 0-100>, \
"explanation": "<string>", "cited_sources": [<integers>]}"""


class FactCheckUnavailableError(RuntimeError):
    """Raised when live fact-checking isn't configured (missing API keys)."""


class FactCheckServiceError(RuntimeError):
    """Raised when the search or LLM upstream call fails."""


@dataclass(frozen=True)
class Source:
    index: int
    title: str
    url: str
    snippet: str
    published_date: str = ""
    cited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_date": self.published_date,
            "cited": self.cited,
        }


@dataclass(frozen=True)
class FactCheckResult:
    claim: str
    verdict: str
    confidence: float
    explanation: str
    sources: list[Source] = field(default_factory=list)
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "sources": [s.to_dict() for s in self.sources],
            "checked_at": self.checked_at,
        }


def _search_web(query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FactCheckServiceError(f"Web search failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise FactCheckServiceError("Web search returned an unreadable response.") from exc

    return data.get("results", [])


def _ask_llm(
    claim: str,
    sources: list[Source],
    today: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    listing = "\n\n".join(
        f"[{s.index}] {s.title} — {s.url}\n"
        f"Published: {s.published_date or 'unknown'}\n"
        f"{s.snippet}"
        for s in sources
    )
    user_prompt = f"TODAY'S DATE: {today}\n\nCLAIM: {claim}\n\nSOURCES:\n\n{listing}"

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FactCheckServiceError(f"Reasoning model call failed: {exc}") from exc

    try:
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError) as exc:
        raise FactCheckServiceError("Reasoning model returned an unreadable response.") from exc

    return parsed


def verify_claim(
    claim: str,
    *,
    tavily_api_key: str,
    groq_api_key: str,
    groq_base_url: str,
    groq_model: str,
    max_sources: int = 5,
) -> FactCheckResult:
    if not tavily_api_key or not groq_api_key:
        raise FactCheckUnavailableError(
            "Live fact-checking isn't configured. Set TAVILY_API_KEY and GROQ_API_KEY."
        )

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d (%A)")

    raw_results = _search_web(claim, tavily_api_key, max_sources)

    if not raw_results:
        return FactCheckResult(
            claim=claim,
            verdict="UNVERIFIED",
            confidence=0.0,
            explanation="No current web sources were found for this claim, so it can't be verified right now.",
            sources=[],
            checked_at=now.isoformat(),
        )

    sources = [
        Source(
            index=i + 1,
            title=(r.get("title") or "Untitled")[:200],
            url=r.get("url", ""),
            snippet=(r.get("content") or "")[:MAX_SNIPPET_CHARS],
            published_date=(r.get("published_date") or "").strip(),
        )
        for i, r in enumerate(raw_results)
    ]

    verdict_raw = _ask_llm(claim, sources, today_str, groq_api_key, groq_base_url, groq_model)

    verdict = str(verdict_raw.get("verdict", "UNVERIFIED")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "UNVERIFIED"

    try:
        confidence = float(verdict_raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(100.0, confidence))

    explanation = str(verdict_raw.get("explanation", "")).strip() or "No explanation returned."

    cited_indices = set()
    for idx in verdict_raw.get("cited_sources", []) or []:
        try:
            cited_indices.add(int(idx))
        except (TypeError, ValueError):
            continue

    sources = [
        Source(s.index, s.title, s.url, s.snippet, s.published_date, cited=s.index in cited_indices)
        for s in sources
    ]

    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        confidence=round(confidence, 1),
        explanation=explanation,
        sources=sources,
        checked_at=now.isoformat(),
    )
