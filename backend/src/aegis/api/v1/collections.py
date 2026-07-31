"""Collection catalogue endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter
from pydantic import TypeAdapter

from aegis.api.deps import ContainerDep, PrincipalDep, ReposDep
from aegis.api.schemas import CollectionCreate, CollectionOut, PageOut
from aegis.core.ids import new_id
from aegis.domain.entities import Collection
from aegis.domain.enums import Permission

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=PageOut)
async def list_collections(
    principal: PrincipalDep,
    repos: ReposDep,
) -> PageOut:
    principal.require(Permission.COLLECTION_READ, resource="collections")
    # Guests and customers only see collections for their mode; admins see all.
    if principal.role.is_internal and principal.has(Permission.COLLECTION_MANAGE):
        rows = await repos.collections.list_all()
    else:
        rows = await repos.collections.list_for_mode(principal.ctx.mode)
    items = TypeAdapter(list[CollectionOut]).validate_python(rows)
    return PageOut(items=items, has_more=False, total_estimate=len(items))


@router.post("", response_model=CollectionOut, status_code=201)
async def create_collection(
    body: CollectionCreate,
    principal: PrincipalDep,
    repos: ReposDep,
    container: ContainerDep,
) -> CollectionOut:
    principal.require(Permission.COLLECTION_MANAGE, resource="collections")
    settings = container.settings
    collection = Collection(
        id=new_id(),
        name=body.name.strip(),
        slug=body.slug.strip().lower(),
        description=body.description,
        mode=body.mode,
        default_visibility=body.default_visibility,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        vector_backend=settings.vector_backend,
        vector_namespace=f"col_{body.slug.strip().lower()}",
        chunk_strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap_tokens,
    )
    created = await repos.collections.create(collection)
    await repos.uow.commit()
    return CollectionOut.model_validate(created)


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(
    collection_id: UUID,
    principal: PrincipalDep,
    repos: ReposDep,
) -> CollectionOut:
    principal.require(Permission.COLLECTION_READ, resource="collections")
    collection = await repos.collections.get(collection_id)
    from aegis.core.errors import NotFoundError

    if collection is None:
        raise NotFoundError("Collection", collection_id)
    if (
        not principal.has(Permission.COLLECTION_MANAGE)
        and collection.mode is not principal.ctx.mode
    ):
        raise NotFoundError("Collection", collection_id)
    return CollectionOut.model_validate(collection)
