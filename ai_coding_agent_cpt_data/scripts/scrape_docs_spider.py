#!/usr/bin/env python3
"""
Scrape & Spider AI Coding Agent documentation.
Targets:
- https://code.claude.com/docs/ja/overview
- https://code.claude.com/docs/en/overview
- https://learn.chatgpt.com/docs

Features:
- Crawls internal links recursively (BFS spidering).
- Leverages llms.txt / sitemap if available for full coverage.
- Extracts clean markdown/text using trafilatura & BeautifulSoup.
- Saves results to data/scraped_documents.jsonl and data/scraped_manifest.csv.
"""

import os
import re
import sys
import time
import json
import csv
from collections import deque
from typing import Set, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import trafilatura

# Configuration
SEEDS = [
    {
        "name": "claude_code_en",
        "url": "https://code.claude.com/docs/en/overview",
        "allowed_domains": ["code.claude.com"],
        "path_prefixes": ["/docs/en/", "/docs/"],
        "llms_txt": "https://code.claude.com/docs/llms.txt"
    },
    {
        "name": "learn_chatgpt",
        "url": "https://learn.chatgpt.com/docs",
        "allowed_domains": ["learn.chatgpt.com", "developers.openai.com"],
        "path_prefixes": ["/docs", "/codex", "/"],
        "llms_txt": "https://learn.chatgpt.com/llms.txt"
    }
]

MAX_PAGES_PER_SEED = 300
REQUEST_DELAY = 0.3  # seconds between requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AI-Coding-Agent-Research/1.0"
}


def normalize_url(url: str) -> str:
    """Normalize URL by stripping fragments and trailing slashes where appropriate."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def is_allowed_url(url: str, allowed_domains: List[str], path_prefixes: List[str]) -> bool:
    """Check if URL belongs to target domains and valid path prefixes."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc not in allowed_domains:
        return False
    
    # Exclude non-document assets
    if re.search(r'\.(png|jpg|jpeg|gif|svg|ico|css|js|woff|woff2|ttf|eot|pdf|zip)$', parsed.path, re.IGNORECASE):
        return False
    
    # Check prefixes
    if any(parsed.path.startswith(prefix) for prefix in path_prefixes):
        return True
    return False


def fetch_url(session: requests.Session, url: str) -> Optional[requests.Response]:
    """Fetch URL with basic error handling and timeouts."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp
        else:
            print(f"[HTTP {resp.status_code}] {url}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def extract_links_from_html(html: str, base_url: str, allowed_domains: List[str], path_prefixes: List[str]) -> List[str]:
    """Extract valid internal links from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    extracted = []
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue
        full_url = urljoin(base_url, href)
        norm_url = normalize_url(full_url)
        if is_allowed_url(norm_url, allowed_domains, path_prefixes):
            extracted.append(norm_url)
    return extracted


def parse_llms_txt(llms_txt_content: str, base_url: str) -> List[str]:
    """Extract URLs listed in llms.txt markdown document."""
    urls = []
    # Match markdown links [Title](URL) or raw URLs
    link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+|\/[^\)]+)\)')
    for match in link_pattern.finditer(llms_txt_content):
        href = match.group(2).strip()
        full_url = urljoin(base_url, href)
        urls.append(normalize_url(full_url))
    return urls


def process_content(url: str, resp: requests.Response) -> Dict:
    """Extract text content, title, and metadata from response."""
    content_type = resp.headers.get("Content-Type", "")
    html_text = resp.text
    
    title = ""
    extracted_text = ""

    # Attempt extraction via trafilatura first
    trafilatura_text = trafilatura.extract(html_text, include_links=False, include_formatting=True)
    
    # BeautifulSoup extraction for title and fallback text
    soup = BeautifulSoup(html_text, 'html.parser')
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.h1:
        title = soup.h1.get_text().strip()
    
    if not title:
        title = url.split('/')[-1] or url

    if trafilatura_text and len(trafilatura_text.strip()) > 50:
        extracted_text = trafilatura_text
    else:
        # Fallback to BeautifulSoup main text extraction
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        main = soup.find('main') or soup.find('article') or soup.body or soup
        extracted_text = main.get_text(separator='\n', strip=True)

    return {
        "url": url,
        "title": title,
        "content": extracted_text,
        "content_length": len(extracted_text),
        "status": resp.status_code
    }


def crawl_seed(seed_info: Dict, session: requests.Session) -> List[Dict]:
    """Run BFS crawl for a specific seed configuration."""
    name = seed_info["name"]
    start_url = normalize_url(seed_info["url"])
    domains = seed_info["allowed_domains"]
    prefixes = seed_info["path_prefixes"]
    llms_txt_url = seed_info.get("llms_txt")

    print(f"\n==========================================")
    print(f" Starting crawl for seed: {name} ({start_url})")
    print(f"==========================================")

    visited: Set[str] = set()
    queue: deque = deque([start_url])
    results: List[Dict] = []

    # Optional: Seed from llms.txt first for high coverage
    if llms_txt_url:
        print(f"[*] Checking llms.txt: {llms_txt_url}")
        resp = fetch_url(session, llms_txt_url)
        if resp:
            llms_urls = parse_llms_txt(resp.text, llms_txt_url)
            print(f"[*] Found {len(llms_urls)} links in llms.txt")
            for u in llms_urls:
                if is_allowed_url(u, domains, prefixes) and u not in visited:
                    queue.append(u)

    while queue and len(results) < MAX_PAGES_PER_SEED:
        url = queue.popleft()
        if url in visited:
            continue

        visited.add(url)
        print(f"[{len(results)+1}/{MAX_PAGES_PER_SEED}] Fetching: {url}")
        
        resp = fetch_url(session, url)
        time.sleep(REQUEST_DELAY)

        if not resp:
            continue

        item = process_content(url, resp)
        item["seed_name"] = name
        results.append(item)

        # Extract next links if HTML
        if "text/html" in resp.headers.get("Content-Type", "").lower() or resp.text.startswith("<!DOCTYPE") or "<html" in resp.text:
            links = extract_links_from_html(resp.text, url, domains, prefixes)
            for link in links:
                if link not in visited:
                    queue.append(link)

    print(f"[+] Finished seed '{name}': Scraped {len(results)} pages.")
    return results


def main():
    os.makedirs("data", exist_ok=True)
    jsonl_output_path = "data/scraped_documents.jsonl"
    csv_output_path = "data/scraped_manifest.csv"
    report_output_path = "data/scraping_report.md"

    session = requests.Session()
    all_documents = []
    
    start_time = time.time()

    for seed in SEEDS:
        seed_results = crawl_seed(seed, session)
        all_documents.extend(seed_results)

    # Save to JSONL
    print(f"\n[*] Writing results to {jsonl_output_path}...")
    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save Manifest CSV
    print(f"[*] Writing manifest to {csv_output_path}...")
    with open(csv_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed_name", "url", "title", "content_length", "status"])
        for doc in all_documents:
            writer.writerow([doc["seed_name"], doc["url"], doc["title"], doc["content_length"], doc["status"]])

    # Generate Summary Report
    elapsed = time.time() - start_time
    total_docs = len(all_documents)
    total_chars = sum(d["content_length"] for d in all_documents)

    seed_counts = {}
    for doc in all_documents:
        s_name = doc["seed_name"]
        seed_counts[s_name] = seed_counts.get(s_name, 0) + 1

    report_md = f"""# Web Scraping & Spidering Collection Report

- **Date & Time**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
- **Total Documents Collected**: {total_docs}
- **Total Characters**: {total_chars:,}
- **Time Elapsed**: {elapsed:.2f} seconds

## Collection Summary by Seed

| Seed Name | Pages Scraped | Target Seed URL |
| :--- | :--- | :--- |
"""
    for s in SEEDS:
        name = s["name"]
        count = seed_counts.get(name, 0)
        report_md += f"| `{name}` | {count} | {s['url']} |\n"

    report_md += "\n## Sample Collected Documents\n\n"
    for doc in all_documents[:10]:
        report_md += f"- **[{doc['title']}]({doc['url']})** (`{doc['seed_name']}`): {doc['content_length']:,} chars\n"

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[✓] Scraping complete! Report saved to {report_output_path}")


if __name__ == "__main__":
    main()
