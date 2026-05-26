.PHONY: dev down ingest test benchmark

dev:
	docker compose up -d
	@echo "Infrastructure started!"
	@echo "Kafka: localhost:9092"
	@echo "Qdrant: http://localhost:6333/dashboard"
	@echo "PostgreSQL: localhost:5432"
	@echo "Redis: localhost:6379"

down:
	docker compose down

ingest:
	python -m ingestion.edgar_client
	python -m ingestion.chunk_pipeline

test:
	pytest tests/ -v