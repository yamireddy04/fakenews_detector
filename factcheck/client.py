"""
Fact-Check API integration layer.
Primary:  Google Fact Check Tools API (free, 1000 req/day)
Fallback: ClaimBuster API + local claim decomposition
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

@dataclass
class FactCheckResult:
    claim: str
    rating: str               
    rating_normalized: str    
    publisher: str
    url: str
    review_date: str
    language: str
    confidence: float        


RATING_NORMALIZATION = {
    # FALSE → FAKE
    "false": ("FAKE", 0.95),
    "mostly false": ("FAKE", 0.80),
    "pants on fire": ("FAKE", 1.00),
    "incorrect": ("FAKE", 0.90),
    "fake": ("FAKE", 0.95),
    "scam": ("FAKE", 0.90),
    "misleading": ("FAKE", 0.70),
    "misinformation": ("FAKE", 0.85),
    # REAL
    "true": ("REAL", 0.95),
    "mostly true": ("REAL", 0.80),
    "correct": ("REAL", 0.90),
    "accurate": ("REAL", 0.90),
    # UNVERIFIED / MIXED
    "half true": ("UNVERIFIED", 0.50),
    "mixed": ("UNVERIFIED", 0.50),
    "unproven": ("UNVERIFIED", 0.40),
    "unverified": ("UNVERIFIED", 0.40),
    "needs context": ("UNVERIFIED", 0.45),
    "satire": ("UNVERIFIED", 0.30),
}


def _normalize_rating(rating: str) -> tuple[str, float]:
    key = rating.lower().strip()
    for k, v in RATING_NORMALIZATION.items():
        if k in key:
            return v
    return ("UNVERIFIED", 0.30)

class DiskCache:
    def __init__(self, cache_dir: str = ".factcheck_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl_hours * 3600

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[list]:
        f = self.cache_dir / (self._key(query) + ".json")
        if f.exists():
            data = json.loads(f.read_text())
            if time.time() - data["ts"] < self.ttl:
                return data["results"]
        return None

    def set(self, query: str, results: list):
        f = self.cache_dir / (self._key(query) + ".json")
        f.write_text(json.dumps({"ts": time.time(), "results": results}))

class GoogleFactCheckClient:
    """
    Wraps the Google Fact Check Tools API.
    Docs: https://developers.google.com/fact-check/tools/api/reference/rest

    Set GOOGLE_API_KEY env var or pass api_key= directly.
    Free tier: 1000 queries/day.
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, api_key: str, cache_dir: str = ".factcheck_cache"):
        self.api_key = api_key
        self.cache = DiskCache(cache_dir)

    def search(
        self,
        query: str,
        language_code: str = "en",
        max_age_days: int = 3650,
        page_size: int = 10,
    ) -> list[FactCheckResult]:
        cached = self.cache.get(query)
        if cached is not None:
            return [FactCheckResult(**r) for r in cached]

        params = {
            "key": self.api_key,
            "query": query,
            "languageCode": language_code,
            "maxAgeDays": max_age_days,
            "pageSize": page_size,
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"

        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except HTTPError as e:
            logger.error(f"Google Fact Check API error {e.code}: {e.reason}")
            return []
        except URLError as e:
            logger.error(f"Network error: {e}")
            return []

        results = self._parse_response(data)
        self.cache.set(query, [asdict(r) for r in results])
        return results

    def _parse_response(self, data: dict) -> list[FactCheckResult]:
        results = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                rating = review.get("textualRating", "Unverified")
                normalized, confidence = _normalize_rating(rating)
                results.append(
                    FactCheckResult(
                        claim=claim.get("text", ""),
                        rating=rating,
                        rating_normalized=normalized,
                        publisher=review.get("publisher", {}).get("name", "Unknown"),
                        url=review.get("url", ""),
                        review_date=review.get("reviewDate", ""),
                        language=review.get("languageCode", "en"),
                        confidence=confidence,
                    )
                )
        return results

class ClaimBusterClient:
    """
    ClaimBuster API — scores sentences for check-worthiness.
    Free tier: https://idir.uta.edu/claimbuster/
    """

    BASE_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def score_claims(self, text: str) -> list[dict]:
        """Returns list of {sentence, score} sorted by check-worthiness."""
        url = f"{self.BASE_URL}{quote_plus(text)}"
        req = Request(url, headers={"x-api-key": self.api_key})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return sorted(
                data.get("results", []),
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
        except Exception as e:
            logger.error(f"ClaimBuster error: {e}")
            return []

class FactChecker:
    """
    Unified interface for fact-checking a claim or article.

    Priority:
      1. Google Fact Check (authoritative external reviews)
      2. ClaimBuster (check-worthiness if no Google result)
    """

    def __init__(
        self,
        google_api_key: str = "",
        claimbuster_api_key: str = "",
        cache_dir: str = ".factcheck_cache",
    ):
        self.google = (
            GoogleFactCheckClient(google_api_key, cache_dir) if google_api_key else None
        )
        self.claimbuster = (
            ClaimBusterClient(claimbuster_api_key) if claimbuster_api_key else None
        )

    def check(self, claim: str, language: str = "en") -> dict:
        """
        Returns:
        {
          "verdict": "FAKE" | "REAL" | "UNVERIFIED" | "NO_DATA",
          "confidence": float,
          "sources": [FactCheckResult, ...],
          "check_worthiness": float | None,
          "summary": str,
        }
        """
        results: list[FactCheckResult] = []

        if self.google:
            results = self.google.search(claim, language_code=language)

        worthiness = None
        if not results and self.claimbuster:
            scored = self.claimbuster.score_claims(claim)
            if scored:
                worthiness = scored[0].get("score", 0)

        if not results:
            return {
                "verdict": "NO_DATA",
                "confidence": 0.0,
                "sources": [],
                "check_worthiness": worthiness,
                "summary": "No fact-check data found for this claim.",
            }

        vote_weights = {"FAKE": 0.0, "REAL": 0.0, "UNVERIFIED": 0.0}
        for r in results:
            vote_weights[r.rating_normalized] += r.confidence

        verdict = max(vote_weights, key=vote_weights.__getitem__)
        total = sum(vote_weights.values()) or 1
        confidence = vote_weights[verdict] / total

        summary_parts = []
        for r in results[:3]:
            summary_parts.append(
                f"{r.publisher} rated this '{r.rating}' ({r.review_date[:10]})"
            )
        summary = "; ".join(summary_parts)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "sources": [asdict(r) for r in results],
            "check_worthiness": worthiness,
            "summary": summary,
        }

    def check_article(self, title: str, body: str = "", language: str = "en") -> dict:
        """
        Extracts key claims from an article and checks each.
        Returns aggregate verdict + per-claim breakdown.
        """
        claims = self._extract_claims(title, body)
        results = [self.check(c, language) for c in claims]

        valid = [r for r in results if r["verdict"] != "NO_DATA"]
        if not valid:
            return {
                "verdict": "NO_DATA",
                "confidence": 0.0,
                "claims_checked": [],
                "summary": "No verifiable claims found.",
            }

        agg = {"FAKE": 0.0, "REAL": 0.0, "UNVERIFIED": 0.0}
        for r in valid:
            agg[r["verdict"]] += r["confidence"]

        verdict = max(agg, key=agg.__getitem__)
        total = sum(agg.values()) or 1

        return {
            "verdict": verdict,
            "confidence": round(agg[verdict] / total, 3),
            "claims_checked": [
                {"claim": c, **r} for c, r in zip(claims, results)
            ],
            "summary": f"Checked {len(valid)} claims. Verdict: {verdict}.",
        }

    @staticmethod
    def _extract_claims(title: str, body: str, max_claims: int = 5) -> list[str]:
        """
        Heuristic claim extractor: title + first sentences with factual markers.
        For production, replace with a dedicated claim-detection model.
        """
        sentences = [title.strip()] if title else []
        for sent in re.split(r"(?<=[.!?])\s+", body):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            if re.search(r"\b(\d+%?|\$[\d,.]+|is|are|was|were|will|has|have)\b", sent):
                sentences.append(sent)
        return sentences[:max_claims]