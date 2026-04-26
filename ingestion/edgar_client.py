import requests
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from bs4 import BeautifulSoup


@dataclass
class Filing:
    company_name: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    document_url: str
    text_content: str = ""


class EdgarClient:
    COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, email: str = "your-email@example.com"):
        self.headers = {
            "User-Agent": f"FinancialIntelligenceSystem {email}",
            "Accept-Encoding": "gzip, deflate",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.data_dir = Path("data/filings")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ticker_to_cik = self._load_ticker_mapping()

    def _rate_limit(self):
        # ugh this rate limit is annoying. sec wants <10 req/s, 0.15 is safe
        time.sleep(0.15)

    def _load_ticker_mapping(self) -> Dict[str, Dict]:
        cache_file = self.cache_dir / "ticker_cik_mapping.json"

        if cache_file.exists():
            import os
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 24:
                with open(cache_file, "r") as f:
                    mapping = json.load(f)
                print(f"ticker map from cache ({len(mapping)} companies)")
                return mapping

        print("downloading sec ticker map...")
        try:
            self._rate_limit()
            resp = self.session.get(self.TICKER_LOOKUP_URL)
            resp.raise_for_status()
            raw_data = resp.json()
        except requests.RequestException as e:
            print(f"ticker map download failed: {e}")
            return {
                "NVDA": {"cik": "0001045810", "name": "NVIDIA CORP"},
                "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
                "MSFT": {"cik": "0000789019", "name": "Microsoft Corporation"},
            }

        mapping = {}
        for entry in raw_data.values():
            ticker = entry.get("ticker", "").upper()
            cik_raw = entry.get("cik_str", "")
            name = entry.get("title", ticker)

            if ticker and cik_raw:
                cik_padded = str(cik_raw).zfill(10)
                mapping[ticker] = {"cik": cik_padded, "name": name}

        with open(cache_file, "w") as f:
            json.dump(mapping, f)

        print(f"loaded {len(mapping)} companies from sec")
        return mapping

    def resolve_ticker(self, ticker: str) -> Optional[Dict]:
        ticker = ticker.upper().strip()
        info = self.ticker_to_cik.get(ticker)
        if not info:
            print(f"ticker '{ticker}' not in sec db")
            return None
        return info

    def get_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ["10-K", "10-Q"],
        max_filings: int = 10
    ) -> List[Dict]:
        info = self.resolve_ticker(ticker)
        if not info:
            return []

        cik = info["cik"]
        company_name = info["name"]

        self._rate_limit()
        try:
            resp = self.session.get(self.COMPANY_URL.format(cik=cik))
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"filing fetch failed for {ticker}: {e}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i in range(len(forms)):
            if forms[i] in filing_types and len(filings) < max_filings:
                acc_clean = accessions[i].replace("-", "")
                filings.append({
                    "company_name": data.get("name", company_name),
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "filing_type": forms[i],
                    "filing_date": dates[i],
                    "accession_number": accessions[i],
                    "document_url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik.lstrip('0')}/{acc_clean}/{primary_docs[i]}"
                    ),
                })
        print(f"found {len(filings)} {filing_types} for {ticker} ({company_name})")
        return filings

    def download_filing_text(self, filing: Dict) -> str:
        filename = f"{filing['ticker']}_{filing['filing_type']}_{filing['filing_date']}.txt"
        filepath = self.data_dir / filename

        if filepath.exists():
            return filepath.read_text(encoding="utf-8")

        self._rate_limit()
        try:
            resp = self.session.get(filing["document_url"])
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"download failed: {e}")
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for el in soup(["script", "style"]):
            el.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        filepath.write_text(text, encoding="utf-8")
        print(f"downloaded: {filename} ({len(text):,} chars)")
        return text

    def ingest_all(self, tickers: List[str]) -> List[Filing]:
        all_filings = []
        for ticker in tickers:
            print(f"\n-> {ticker}")
            for meta in self.get_company_filings(ticker):
                text = self.download_filing_text(meta)
                if text:
                    all_filings.append(Filing(
                        company_name=meta["company_name"],
                        ticker=meta["ticker"],
                        cik=meta["cik"],
                        filing_type=meta["filing_type"],
                        filing_date=meta["filing_date"],
                        accession_number=meta["accession_number"],
                        document_url=meta["document_url"],
                        text_content=text,
                    ))
        print(f"\ntotal: {len(all_filings)} filings")
        return all_filings

    def search_company(self, query: str) -> List[Dict]:
        query_lower = query.lower()
        matches = []
        for ticker, info in self.ticker_to_cik.items():
            if (query_lower in ticker.lower() or
                query_lower in info["name"].lower()):
                matches.append({
                    "ticker": ticker,
                    "name": info["name"],
                    "cik": info["cik"],
                })
        matches.sort(key=lambda x: len(x["name"]))
        return matches[:10]


if __name__ == "__main__":
    client = EdgarClient(email="your-email@example.com")

    print("\nsearch 'tesla':")
    for match in client.search_company("tesla"):
        print(f"  {match['ticker']} - {match['name']}")

    print("\ningesting...")
    filings = client.ingest_all(["TSLA", "NVDA", "GOOGL"])

    for f in filings:
        print(f"  {f.ticker} | {f.filing_type} | {f.filing_date} | {len(f.text_content):,} chars")
