"""Typer CLI for administrative bootstrap tasks."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer

from aegis.core.config import get_settings
from aegis.core.container import Container
from aegis.core.ids import new_id
from aegis.core.logging import configure_logging, get_logger
from aegis.core.security import hash_password, validate_password_strength
from aegis.domain.entities import Collection
from aegis.domain.enums import Mode, Role, Visibility
from aegis.domain.policies.permissions import DEFAULT_ROLE_PERMISSIONS

app = typer.Typer(name="aegis", help="Aegis RAG platform administration.", no_args_is_help=True)
logger = get_logger(__name__)


def _run(coro: object) -> None:
    asyncio.run(coro)  # type: ignore[arg-type]


@app.command("create-admin")
def create_admin(
    email: Annotated[str, typer.Option(help="Admin email address")],
    password: Annotated[str, typer.Option(help="Initial password")],
    full_name: Annotated[str, typer.Option("--name", help="Display name")] = "Administrator",
) -> None:
    """Create an admin user (or update password if the email already exists)."""
    _run(_create_admin(email=email, password=password, full_name=full_name))


async def _create_admin(*, email: str, password: str, full_name: str) -> None:
    settings = get_settings()
    configure_logging(settings)
    validate_password_strength(password, settings, email=email)
    container = Container.build(settings)
    await container.startup()
    try:
        async with container.session() as (_session, repos):
            pepper = settings.secret_key.get_secret_value()
            existing = await repos.users.get_by_email(email.strip().lower())
            if existing is not None:
                await repos.users.set_password(
                    existing.id, hash_password(password, pepper=pepper)
                )
                await repos.users.update(existing.id, role=Role.ADMIN, is_active=True)
                await repos.uow.commit()
                typer.echo(f"Updated existing user {existing.email} to admin.")
                return
            user = await repos.users.create(
                email=email.strip().lower(),
                full_name=full_name.strip(),
                role=Role.ADMIN,
                password_hash=hash_password(password, pepper=pepper),
                must_change_password=False,
            )
            await repos.uow.commit()
            typer.echo(f"Created admin {user.email} ({user.id})")
    finally:
        await container.shutdown()


@app.command("seed")
def seed(
    admin_email: Annotated[
        str, typer.Option("--admin-email", help="Demo admin email")
    ] = "admin@aegis.local",
    admin_password: Annotated[
        str, typer.Option("--admin-password", help="Demo admin password")
    ] = "ChangeMe-Admin-2026!",  # noqa: S107 - local demo seed default only
) -> None:
    """Seed role permissions, an admin user, and sample collections."""
    _run(_seed(admin_email=admin_email, admin_password=admin_password))


async def _seed(*, admin_email: str, admin_password: str) -> None:
    settings = get_settings()
    configure_logging(settings)
    container = Container.build(settings)
    await container.startup()
    try:
        async with container.session() as (_session, repos):
            await repos.permissions.seed_defaults(DEFAULT_ROLE_PERMISSIONS)

            pepper = settings.secret_key.get_secret_value()
            validate_password_strength(admin_password, settings, email=admin_email)
            if await repos.users.get_by_email(admin_email.strip().lower()) is None:
                await repos.users.create(
                    email=admin_email.strip().lower(),
                    full_name="Aegis Admin",
                    role=Role.ADMIN,
                    password_hash=hash_password(admin_password, pepper=pepper),
                )
                typer.echo(f"Created admin {admin_email}")
            else:
                typer.echo(f"Admin {admin_email} already exists")

            for name, slug, mode, visibility in (
                ("Internal Knowledge Base", "internal-kb", Mode.INTERNAL, Visibility.INTERNAL),
                ("Customer Help Centre", "customer-help", Mode.CUSTOMER, Visibility.CUSTOMER),
            ):
                existing = await repos.collections.get_by_slug(slug)
                if existing is not None:
                    typer.echo(f"Collection {slug} already exists")
                    continue
                await repos.collections.create(
                    Collection(
                        id=new_id(),
                        name=name,
                        slug=slug,
                        description=f"Demo {mode.value} collection",
                        mode=mode,
                        default_visibility=visibility,
                        embedding_provider=settings.embedding_provider,
                        embedding_model=settings.embedding_model,
                        embedding_dim=settings.embedding_dim,
                        vector_backend=settings.vector_backend,
                        vector_namespace=f"col_{slug.replace('-', '_')}",
                        chunk_strategy=settings.chunk_strategy,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap_tokens,
                    )
                )
                typer.echo(f"Created collection {slug}")

            await repos.uow.commit()
            typer.echo("Seed complete.")
    finally:
        await container.shutdown()


if __name__ == "__main__":
    app()
