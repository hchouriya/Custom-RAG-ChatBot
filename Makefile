.PHONY: up down infra migrate seed logs build ps restart

# Full stack (infra + api + worker + frontend)
up:
	docker compose up -d --build

# Infrastructure only (postgres, redis, qdrant, minio)
infra:
	docker compose up -d postgres redis qdrant minio minio-init

down:
	docker compose down

build:
	docker compose build

migrate:
	docker compose run --rm api alembic upgrade head

seed:
	docker compose run --rm api aegis seed

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

restart:
	docker compose restart api worker frontend
