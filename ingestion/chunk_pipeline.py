import json, pickle, uuid
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from config import settings
from ingestion.edgar_client import EdgarClient, Filing


class ChunkPipeline:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        # x4 because chunk_size is tokens-ish, splitter wants chars
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE * 4,
            chunk_overlap=settings.CHUNK_OVERLAP * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.bm25_data_path = Path("data/bm25_index.pkl")
        self.bm25_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_collection()

    def _ensure_collection(self):
        names = [c.name for c in self.qdrant.get_collections().collections]
        if settings.QDRANT_COLLECTION_NAME not in names:
            self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
            print(f"created qdrant collection: {settings.QDRANT_COLLECTION_NAME}")

    def _parse_fiscal_year(self, filing_date: str, filing_type: str) -> int:
        try:
            year = int(filing_date[:4])
            month = int(filing_date[5:7])
        except (ValueError, IndexError):
            return 0

        if filing_type in ("10-K", "10-K/A") and month <= 3:
            return year - 1
        return year

    def _parse_fiscal_quarter(self, filing_date: str, filing_type: str) -> str:
        if filing_type in ("10-K", "10-K/A"):
            return "FY"

        try:
            month = int(filing_date[5:7])
        except (ValueError, IndexError):
            return "unknown"

        # hack: overlapping ranges, picks first match. close enough for filing dates
        if month in (2, 3, 4, 5):
            return "Q1"
        elif month in (5, 6, 7, 8):
            return "Q2"
        elif month in (8, 9, 10, 11):
            return "Q3"
        else:
            return "Q4"

    def chunk_filing(self, filing: Filing) -> List[Dict]:
        raw_chunks = self.splitter.split_text(filing.text_content)

        fiscal_year = self._parse_fiscal_year(filing.filing_date, filing.filing_type)
        fiscal_quarter = self._parse_fiscal_quarter(filing.filing_date, filing.filing_type)

        chunks = []
        for i, text in enumerate(raw_chunks):
            if len(text.strip()) < 50:
                continue
            chunks.append({
                "chunk_id": f"{filing.ticker}_{filing.filing_type}_{filing.filing_date}_chunk_{i:04d}",
                "text": text.strip(),
                "ticker": filing.ticker,
                "company_name": filing.company_name,
                "filing_type": filing.filing_type,
                "filing_date": filing.filing_date,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "chunk_index": i,
                "source_doc": f"{filing.ticker}_{filing.filing_type}_{filing.filing_date}",
            })
        print(f"  {filing.ticker} {filing.filing_type} ({filing.filing_date}) "
              f"-> fy={fiscal_year} {fiscal_quarter} "
              f"-> {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        texts = [c["text"] for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            resp = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL, input=batch,
            )
            all_embeddings.extend([item.embedding for item in resp.data])
            print(f"  embed batch {i//100 + 1}/{(len(texts)-1)//100 + 1}")
        return all_embeddings

    def index_to_qdrant(self, chunks: List[Dict], embeddings: List[List[float]]):
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=emb, payload=chunk)
            for chunk, emb in zip(chunks, embeddings)
        ]
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.qdrant.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=batch)
            print(f"  upsert {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        print(f"  indexed {len(points)} chunks in qdrant")

    def build_bm25_index(self, chunks: List[Dict]):
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with open(self.bm25_data_path, "wb") as f:
            pickle.dump({"index": bm25, "chunks": chunks}, f)
        print(f"  bm25 built ({len(chunks)} chunks)")

    def patch_fiscal_years_in_qdrant(self):
        print("\npatching fiscal_year in qdrant...")
        offset = None
        total_patched = 0

        while True:
            results, next_offset = self.qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                scroll_filter=None,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            points_to_update = []
            for point in results:
                payload = point.payload
                filing_date = payload.get("filing_date", "")
                filing_type = payload.get("filing_type", "")
                current_year = payload.get("fiscal_year", 0)

                if filing_date and (current_year == 0 or current_year is None):
                    new_year = self._parse_fiscal_year(filing_date, filing_type)
                    new_quarter = self._parse_fiscal_quarter(filing_date, filing_type)
                    points_to_update.append((point.id, new_year, new_quarter))

            if points_to_update:
                for point_id, new_year, new_quarter in points_to_update:
                    self.qdrant.set_payload(
                        collection_name=settings.QDRANT_COLLECTION_NAME,
                        payload={"fiscal_year": new_year, "fiscal_quarter": new_quarter},
                        points=[point_id],
                    )
                total_patched += len(points_to_update)
                print(f"  patched {total_patched} so far...")

            offset = next_offset
            if offset is None:
                break

        print(f"\ndone. patched {total_patched} chunks.")

    def process_all(self, filings: List[Filing]):
        all_chunks = []
        print("\nchunking...")
        for filing in filings:
            all_chunks.extend(self.chunk_filing(filing))
        print(f"\ntotal chunks: {len(all_chunks)}")

        if not all_chunks:
            return

        print("\nembedding...")
        embeddings = self.embed_chunks(all_chunks)

        print("\nqdrant...")
        self.index_to_qdrant(all_chunks, embeddings)

        print("\nbm25...")
        self.build_bm25_index(all_chunks)

        print(f"\ndone. {len(all_chunks)} chunks indexed.")


if __name__ == "__main__":
    edgar = EdgarClient(email="your-email@example.com")
    filings = edgar.ingest_all(["NVDA", "AAPL", "MSFT"])
    ChunkPipeline().process_all(filings)
