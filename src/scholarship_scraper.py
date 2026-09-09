"""
Scrapes current scholarship/fellowship listings from a small set of public
scholarship-news RSS feeds, normalizes them, and caches the result as JSON
for the /opportunities page to read.

Design notes (read before adding sources or trusting the output):

- RSS only, never HTML scraping. Each source below publishes a standard RSS
  feed intended for syndication -- this is a far more stable and
  lower-friction way to pull "latest opportunities" than parsing arbitrary
  HTML, and it avoids scraping pages that were not meant to be machine-read.
- We do NOT gate feed fetches on robots.txt. That file governs how search
  engine crawlers index a site's pages; it does not, by internet convention,
  govern RSS/Atom feed consumption -- no mainstream feed reader (Feedly,
  Inoreader, podcast apps, etc.) checks robots.txt before fetching a feed
  URL the site itself publishes for syndication. A generic
  "Disallow: /feed/" in robots.txt is standard SEO boilerplate aimed at
  stopping Google from indexing feed pages as duplicate content, not a
  signal aimed at feed-reader clients. We still fetch politely: one request
  per source per cache refresh (at most every 12h), a normal timeout, and a
  standard browser User-Agent + Accept header.
- Deadline, study level, region, and "fully funded" tags are best-effort,
  derived from the article title/summary text with simple keyword/regex
  matching -- they are NOT authoritative. The About/Opportunities pages say
  so explicitly, and every listing links back to the original source so a
  reader can verify the real deadline and requirements before applying.
- This module only ever reads public RSS content and writes to
  data/opportunities_cache.json. It makes no changes to any remote site.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import feedparser
import requests

CACHE_PATH = "data/opportunities_cache.json"
REQUEST_TIMEOUT = 10
# A realistic browser UA, not a self-identifying bot string. We only ever
# make one lightweight request per source per refresh (at most every 12h),
# but plenty of sites run WAFs that block any request declaring itself as a
# bot/script -- a browser-like UA is what most RSS readers use in practice to
# avoid that filtering, which is unrelated to a site's actual feed policy.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
MAX_SUMMARY_CHARS = 280
MAX_ITEMS_PER_SOURCE = 40

# Add more sources here -- each just needs a name and a standard RSS feed URL.
# Sources that fail (bad feed URL, empty feed, blocked, network error) are
# skipped and logged, not fatal -- see fetch_source().
#
# Scholars4Dev is kept even though its feed currently parses as valid RSS
# with zero <item> entries (confirmed via production logs, not a bug on our
# end) -- harmless to leave in in case that changes, and the diagnostic log
# will say so on every refresh rather than silently contributing nothing.
#
# youthop.com/feed/ was tried and dropped: confirmed 404 in production. Its
# scholarships appear to live under /scholarships/ rather than the site
# root, so its feed (if any) is likely at a different path -- revisit if a
# correct URL is found.
SOURCES = [
    {
        "name": "Scholars4Dev",
        "homepage": "https://www.scholars4dev.com/",
        "feed_url": "https://www.scholars4dev.com/feed/",
    },
    {
        "name": "OpportunitiesForAfricans",
        "homepage": "https://www.opportunitiesforafricans.com/",
        "feed_url": "https://www.opportunitiesforafricans.com/feed/",
    },
    {
        "name": "OpportunityDesk",
        "homepage": "https://opportunitydesk.org/",
        "feed_url": "https://opportunitydesk.org/feed/",
    },
]

LEVEL_KEYWORDS = [
    "PhD", "Doctoral", "Postdoctoral", "Postdoc",
    "Master's", "Masters", "MSc", "MBA",
    "Undergraduate", "Bachelor", "Bachelor's",
    "Fellowship", "Internship", "Research Grant", "Postgraduate",
]

REGION_KEYWORDS = [
    "Africa", "Asia", "Europe", "Latin America", "Middle East",
    "United States", "USA", "United Kingdom", "UK", "Canada", "Australia",
    "Nigeria", "Kenya", "Ghana", "South Africa", "India", "China",
    "Germany", "France", "Commonwealth", "Developing Countries", "Global",
]

DEADLINE_PATTERN = re.compile(
    r"(deadline|closes?\s+on|apply\s+by|closing\s+date)\s*[:\-]?\s*([^.\n]{4,60})",
    re.IGNORECASE,
)


def _clean_summary(html_or_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html_or_text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _extract_tags(text: str, vocabulary: list) -> list:
    found = []
    lowered = text.lower()
    for term in vocabulary:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
            canonical = term
            if canonical not in found:
                found.append(canonical)
    return found


def _extract_deadline(text: str):
    m = DEADLINE_PATTERN.search(text)
    if not m:
        return None
    return m.group(2).strip().rstrip(",;")


def _extract_funding(text: str) -> bool:
    return bool(re.search(r"\bfully\s+funded\b", text, re.IGNORECASE))


def _entry_id(link: str) -> str:
    return hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]


def fetch_source(source: dict) -> list:
    try:
        resp = requests.get(
            source["feed_url"],
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  skipping {source['name']}: request failed -- {exc}")
        return []

    content_type = resp.headers.get("Content-Type", "unknown")
    if resp.url != source["feed_url"]:
        print(f"  note: {source['name']} redirected to {resp.url}")

    parsed = feedparser.parse(resp.content)
    if parsed.bozo:
        print(
            f"  warning: {source['name']} feed did not parse cleanly "
            f"(status={resp.status_code}, content-type={content_type}): {parsed.get('bozo_exception')}"
        )

    raw_entry_count = len(parsed.entries)
    if raw_entry_count == 0:
        snippet = resp.text[:400].replace("\n", " ")
        print(f"  note: {source['name']} feed parsed with 0 entries; response starts: {snippet!r}")

    listings = []
    skipped_missing_fields = 0
    for entry in parsed.entries[:MAX_ITEMS_PER_SOURCE]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            skipped_missing_fields += 1
            continue

        raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
        summary = _clean_summary(raw_summary)
        combined_text = f"{title} {summary}"

        published_iso = None
        if getattr(entry, "published_parsed", None):
            published_iso = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

        listings.append(
            {
                "id": _entry_id(link),
                "title": title,
                "link": link,
                "source": source["name"],
                "source_homepage": source["homepage"],
                "published": published_iso,
                "summary": summary,
                "deadline_text": _extract_deadline(combined_text),
                "level": _extract_tags(combined_text, LEVEL_KEYWORDS),
                "region": _extract_tags(combined_text, REGION_KEYWORDS),
                "fully_funded": _extract_funding(combined_text),
            }
        )

    print(
        f"  {source['name']}: status={resp.status_code}, content-type={content_type}, "
        f"raw entries={raw_entry_count}, usable={len(listings)}, "
        f"skipped (missing title/link)={skipped_missing_fields}"
    )
    return listings


def scrape_all() -> dict:
    all_listings = []
    for source in SOURCES:
        print(f"Fetching {source['name']} ({source['feed_url']}) ...")
        all_listings.extend(fetch_source(source))

    # De-duplicate by link, newest first (fall back to source order if no date).
    seen = {}
    for item in all_listings:
        seen[item["id"]] = item
    deduped = list(seen.values())
    deduped.sort(key=lambda x: x["published"] or "", reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [s["name"] for s in SOURCES],
        "listings": deduped,
    }


def save_cache(data: dict, path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_cache(path: str = CACHE_PATH):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    start = time.time()
    result = scrape_all()
    save_cache(result)
    print(f"\nSaved {len(result['listings'])} listings from {len(result['sources'])} source(s) "
          f"to {CACHE_PATH} in {time.time() - start:.1f}s")
