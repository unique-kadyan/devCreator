"""Topic collection from sources that are genuinely free and genuinely allowed.

What is in and why:
  * rss        - public feeds. No key, no quota beyond politeness. Always available.
  * wikipedia  - the public REST API. No key. Rate limit is generous; we send a real
                 User-Agent because the WMF asks contactable clients to identify themselves.
  * seasonal   - a local calendar. No network at all, so research never fails completely.
  * youtube_search - 100 units per call against a 10,000/day budget. Capped in config.

What is deliberately out:
  * reddit  - self-service app registration closed in late 2025 and approval now takes
              weeks. It is wired as opt-in, disabled by default, and never assumed.
  * pytrends - the library was archived on 2025-04-17 and the official Trends API is an
              application-gated alpha. Nothing here depends on trend data.

Nothing in this module scrapes a site that forbids it, and nothing pretends a paid API is
free.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..core.errors import ProviderError, QuotaExhausted, RateLimited
from ..core.logging import get_logger

log = get_logger("research")

UA = "animal-story-automation/0.1 (personal project; contact via channel about page)"
ANIMALS = ["fox", "rabbit", "lion", "cat", "monkey", "dog", "bear", "mouse", "owl",
           "goat", "raccoon", "hedgehog"]
ARCHETYPES = ["underdog", "trickster", "redemption", "mystery", "friendship", "survival",
              "comedy", "family"]

DEFAULT_FEEDS = [
    "https://www.sciencedaily.com/rss/plants_animals/animals.xml",
    "https://phys.org/rss-feed/biology-news/plants-animals/",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
]


@dataclass
class Candidate:
    topic: str
    keywords: list[str] = field(default_factory=list)
    primary_animal: str | None = None
    archetype: str | None = None
    source: str = "manual"
    source_ref: str | None = None
    signals: dict = field(default_factory=dict)


def _animal_in(text: str) -> str | None:
    low = text.lower()
    for a in ANIMALS:
        if re.search(rf"\b{a}(es|s)?\b", low):
            return a
    return None


# Three letters, not four: "fox", "cat", "dog" and "owl" are the most load-bearing words
# this pipeline can see, and a 4-character floor drops every one of them.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was",
    "were", "are", "its", "his", "her", "hers", "their", "they", "them", "you", "your",
    "how", "why", "what", "when", "who", "new", "says", "said", "about", "into", "than",
    "but", "not", "can", "all", "one", "two", "out", "did", "get", "got", "see", "say",
    "she", "him", "our", "his", "off", "own", "too", "any", "may", "now", "way", "use",
    "over", "just", "only", "also", "more", "most", "some", "such", "then", "there",
    "here", "been", "being", "will", "would", "could", "should", "after", "before",
}


def _keywords(text: str, n: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------- RSS

def collect_rss(feeds: list[str] | None = None, per_feed: int = 8,
                timeout_s: float = 20.0) -> list[Candidate]:
    out: list[Candidate] = []
    for url in (feeds or DEFAULT_FEEDS):
        try:
            r = httpx.get(url, timeout=timeout_s, headers={"User-Agent": UA},
                          follow_redirects=True)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except (httpx.HTTPError, ET.ParseError) as e:
            log.warning("rss_failed", feed=url, error=str(e)[:140])
            continue
        items = root.findall(".//item") or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry")
        for item in items[:per_feed]:
            title = (item.findtext("title")
                     or item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            if not title:
                continue
            link = (item.findtext("link")
                    or item.findtext("{http://www.w3.org/2005/Atom}id") or url)
            out.append(Candidate(topic=title, keywords=_keywords(title),
                                 primary_animal=_animal_in(title), source="rss",
                                 source_ref=link, signals={"freshness": 1.0}))
    log.info("rss_collected", n=len(out))
    return out


# ------------------------------------------------------------------ Wikipedia

def collect_wikipedia(seeds: list[str] | None = None, timeout_s: float = 20.0
                      ) -> list[Candidate]:
    """Pull a plain-language summary for each animal and mine it for story hooks.

    Animal facts are a good seed because they are true, they are specific, and nobody owns
    them - the opposite of building stories out of somebody else's characters.
    """
    seeds = seeds or ANIMALS
    out: list[Candidate] = []
    with httpx.Client(timeout=timeout_s, headers={"User-Agent": UA},
                      follow_redirects=True) as client:
        for animal in seeds:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{animal}"
            try:
                r = client.get(url)
                if r.status_code == 429:
                    raise RateLimited("wikipedia rate limited", provider="wikipedia")
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                log.warning("wikipedia_failed", animal=animal, error=str(e)[:140])
                continue
            extract = (data.get("extract") or "").strip()
            if not extract:
                continue
            # First sentence after the definition tends to hold the interesting behaviour.
            sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", extract) if s.strip()]
            for s in sentences[1:3]:
                out.append(Candidate(
                    topic=f"A {animal} story inspired by: {s[:180]}",
                    keywords=_keywords(s), primary_animal=animal, source="wikipedia",
                    source_ref=data.get("content_urls", {}).get("desktop", {}).get("page"),
                    signals={"factual": 1.0}))
    log.info("wikipedia_collected", n=len(out))
    return out


# ------------------------------------------------------------------- seasonal

SEASONAL = {
    1: ("fresh starts", "a promise made in the cold"), 2: ("friendship", "an unlikely pair"),
    3: ("first green", "something buried starts to grow"),
    4: ("rain and mud", "a plan ruined by weather"),
    5: ("markets and making", "a stall that nobody visits"),
    6: ("long evenings", "a secret kept all summer"),
    7: ("heat and drought", "a well that runs dry"),
    8: ("harvest coming", "too much work, too few hands"),
    9: ("back to work", "the new one who does not fit"),
    10: ("dark evenings", "a sound in the woods with an ordinary cause"),
    11: ("storing up", "sharing when there is not enough"),
    12: ("giving", "a gift that costs the giver something"),
}


def collect_seasonal(today: dt.date | None = None) -> list[Candidate]:
    today = today or dt.date.today()
    out = []
    for month in {today.month, (today.month % 12) + 1}:
        theme, hook = SEASONAL[month]
        for animal in ANIMALS[:6]:
            out.append(Candidate(
                topic=f"{hook} - a {animal} story about {theme}",
                keywords=[theme.split()[0], animal, "seasonal"],
                primary_animal=animal, source="seasonal",
                source_ref=f"month:{month}", signals={"seasonal": 1.0}))
    log.info("seasonal_collected", n=len(out))
    return out


# -------------------------------------------------------------- YouTube search

def collect_youtube_search(api_key: str, queries: list[str], quota=None,
                           max_calls: int = 4, timeout_s: float = 25.0) -> list[Candidate]:
    """search.list costs 100 units of a 10,000/day budget - hence the hard call cap.

    This reads the landscape (what already exists, how crowded a niche is). It never copies
    a title, a thumbnail or a premise; competition is a signal, not a template.
    """
    if not api_key:
        log.info("youtube_search_skipped", reason="no YT_API_KEY")
        return []
    out: list[Candidate] = []
    calls = 0
    with httpx.Client(timeout=timeout_s) as client:
        for q in queries[:max_calls]:
            if calls >= max_calls:
                break
            if quota is not None:
                from ..core.quota import Limits
                try:
                    quota.check("youtube", Limits(units_per_day=10_000), units=100)
                except QuotaExhausted:
                    log.warning("youtube_search_quota_stop", after_calls=calls)
                    break
            try:
                r = client.get("https://www.googleapis.com/youtube/v3/search", params={
                    "part": "snippet", "q": q, "type": "video", "maxResults": 10,
                    "order": "viewCount", "relevanceLanguage": "en",
                    "publishedAfter": (dt.datetime.now(dt.UTC) - dt.timedelta(days=120)
                                       ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "key": api_key})
            except httpx.HTTPError as e:
                raise ProviderError(f"youtube search failed: {e}",
                                    provider="youtube") from e
            calls += 1
            if quota is not None:
                quota.consume("youtube", units=100)
            if r.status_code == 403:
                raise QuotaExhausted("youtube search quota or key rejected",
                                     provider="youtube")
            if r.status_code >= 400:
                log.warning("youtube_search_http", status=r.status_code, query=q)
                continue
            items = r.json().get("items", [])
            competition = min(1.0, len(items) / 10.0)
            for it in items[:4]:
                title = it["snippet"]["title"]
                out.append(Candidate(
                    topic=f"An original animal story in the space of: {q}",
                    keywords=_keywords(q + " " + title), primary_animal=_animal_in(title),
                    source="youtube_search",
                    source_ref=f"https://youtu.be/{it['id']['videoId']}",
                    signals={"search_demand": 1.0, "competition": competition}))
    log.info("youtube_search_collected", n=len(out), calls=calls, units=calls * 100)
    return out


def collect_all(cfg, quota=None) -> list[Candidate]:
    enabled = set(cfg.get("research.collectors", ["rss", "wikipedia", "seasonal"]))
    out: list[Candidate] = []
    if "rss" in enabled:
        out += collect_rss(cfg.get("research.feeds"))
    if "wikipedia" in enabled:
        out += collect_wikipedia()
    if "seasonal" in enabled:
        out += collect_seasonal()
    if "youtube_search" in enabled:
        out += collect_youtube_search(
            cfg.secret("YT_API_KEY", required=False),
            cfg.get("research.queries", [
                "animated animal short story", "original animation short film animals",
                "cartoon fox story", "animated fable short"]),
            quota=quota,
            max_calls=int(cfg.get("research.youtube_search_calls_per_day", 4)))
    if "reddit" in enabled:
        log.warning("reddit_collector_requested_but_unavailable",
                    reason="self-service app registration closed; manual approval needed")
    return out
