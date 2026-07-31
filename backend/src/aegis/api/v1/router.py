"""Aggregate v1 router."""

from __future__ import annotations

from fastapi import APIRouter

from aegis.api.v1 import auth, chat, collections, documents

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(collections.router)
