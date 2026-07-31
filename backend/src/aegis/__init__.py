"""Aegis RAG Platform.

An enterprise retrieval-augmented generation service that answers only from company
documents, cites every claim, and enforces permissions before retrieval.

Layering (enforced in CI by import-linter, see pyproject.toml):

    api → services → domain ← infrastructure / rag / agents

``domain`` imports nothing from the project and nothing from a driver, client, or web
framework. Read ``docs/architecture/02-project-structure.md`` before adding a module.
"""

__version__ = "0.2.0"
