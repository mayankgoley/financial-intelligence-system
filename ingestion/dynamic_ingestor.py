import re
from typing import Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import settings


class DynamicIngestor:
    def __init__(self):
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

    def company_exists(self, ticker: str) -> bool:
        ticker = ticker.upper().strip()
        try:
            result = self.qdrant.count(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="ticker",
                            match=MatchValue(value=ticker),
                        )
                    ]
                ),
            )
            count = result.count
            if count > 0:
                print(f"{ticker}: {count} chunks indexed")
                return True
            else:
                print(f"{ticker}: no data, will auto-ingest")
                return False
        except Exception as e:
            print(f"check failed for {ticker}: {e}")
            return False

    def auto_ingest(self, ticker: str) -> bool:
        ticker = ticker.upper().strip()
        print(f"\nauto-ingesting: {ticker}")

        try:
            from ingestion.edgar_client import EdgarClient
            from ingestion.chunk_pipeline import ChunkPipeline

            client = EdgarClient()
            filings = client.ingest_all([ticker])

            if not filings:
                print(f"no filings for {ticker}")
                return False

            pipeline = ChunkPipeline()
            pipeline.process_all(filings)

            print(f"auto-ingest done for {ticker}")
            return True

        except Exception as e:
            print(f"auto-ingest failed for {ticker}: {e}")
            return False

    def ensure_company_data(self, ticker: str) -> Tuple[bool, str]:
        ticker = ticker.upper().strip()

        if not ticker:
            return False, "no ticker"

        if self.company_exists(ticker):
            return True, f"{ticker} ready ({self._get_chunk_count(ticker)} chunks)"

        print(f"{ticker} missing, starting auto-ingest...")
        success = self.auto_ingest(ticker)

        if success and self.company_exists(ticker):
            count = self._get_chunk_count(ticker)
            return True, f"{ticker} ingested ({count} chunks)"
        else:
            return False, f"could not ingest {ticker}. is it a real US ticker?"

    def _get_chunk_count(self, ticker: str) -> int:
        try:
            result = self.qdrant.count(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))]
                ),
            )
            return result.count
        except Exception:
            return 0


# hack: hardcoded for now, fix later — should probably read from sec ticker list
COMPANY_NAMES = {
    "NVIDIA": "NVDA", "APPLE": "AAPL", "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "AMAZON": "AMZN",
    "TESLA": "TSLA", "META": "META", "FACEBOOK": "META",
    "NETFLIX": "NFLX", "AMD": "AMD", "INTEL": "INTC",
    "JPMORGAN": "JPM", "GOLDMAN": "GS", "BERKSHIRE": "BRK-B",
    "VISA": "V", "MASTERCARD": "MA", "DISNEY": "DIS",
    "WALMART": "WMT", "COCA-COLA": "KO", "PEPSI": "PEP",
    "JOHNSON": "JNJ", "PFIZER": "PFE", "UNITEDHEALTH": "UNH",
    "EXXON": "XOM", "CHEVRON": "CVX", "SALESFORCE": "CRM",
    "ADOBE": "ADBE", "ORACLE": "ORCL", "CISCO": "CSCO",
    "QUALCOMM": "QCOM", "BROADCOM": "AVGO", "PAYPAL": "PYPL",
    "UBER": "UBER", "AIRBNB": "ABNB", "SNAP": "SNAP",
    "SPOTIFY": "SPOT", "SHOPIFY": "SHOP", "SQUARE": "SQ",
    "BLOCK": "SQ", "PALANTIR": "PLTR", "SNOWFLAKE": "SNOW",
    "DATADOG": "DDOG", "CROWDSTRIKE": "CRWD", "ZOOM": "ZM",
    "ROBINHOOD": "HOOD", "COINBASE": "COIN",
}


def extract_ticker_from_query(query: str) -> Optional[str]:
    query_upper = query.upper()

    for name, ticker in COMPANY_NAMES.items():
        if name in query_upper:
            return ticker

    words = query_upper.split()
    for word in words:
        clean = re.sub(r'[^A-Z]', '', word)
        if 1 <= len(clean) <= 5 and clean.isalpha():
            if clean in COMPANY_NAMES.values():
                return clean

    return None


if __name__ == "__main__":
    test_queries = [
        "Analyze NVIDIA Q4 earnings",
        "What is Tesla's revenue growth?",
        "How is AAPL performing?",
        "Compare Google and Microsoft cloud revenue",
        "Tell me about Amazon's AWS segment",
    ]
    for q in test_queries:
        ticker = extract_ticker_from_query(q)
        print(f"  '{q}' -> {ticker}")

    ingestor = DynamicIngestor()
    ready, msg = ingestor.ensure_company_data("NVDA")
    print(f"\n{msg}")
