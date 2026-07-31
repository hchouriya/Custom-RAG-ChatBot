-- Extensions required by the Aegis schema (docs/architecture/03).
-- Applied automatically on first Postgres boot via /docker-entrypoint-initdb.d.
-- The pgvector/pgvector:pg16 image already ships the `vector` extension binaries.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
