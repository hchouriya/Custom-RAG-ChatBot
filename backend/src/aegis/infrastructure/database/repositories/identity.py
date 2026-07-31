"""Identity repositories: users, departments, sessions, API keys, permissions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegis.core.errors import NotFoundError
from aegis.core.ids import new_id
from aegis.domain.entities import ApiKey, Department, User, UserCredentials
from aegis.domain.enums import Permission, Role
from aegis.domain.policies.permissions import PERMISSION_DESCRIPTIONS, PermissionOverride
from aegis.infrastructure.database.models import (
    ApiKeyModel,
    DepartmentModel,
    PermissionModel,
    RefreshTokenModel,
    RolePermissionModel,
    UserModel,
    UserPermissionOverrideModel,
)
from aegis.infrastructure.database.repositories.helpers import affected, ltree


def _to_user(row: UserModel) -> User:
    """Map a user row, resolving the department path from the eager-loaded relationship.

    ``department_path`` matters because it is what the ACL policy uses for confidential
    subtree checks; carrying it on the entity avoids a second query per authorization.
    """
    return User(
        id=row.id,
        email=row.email,
        full_name=row.full_name,
        role=row.role,
        department_id=row.department_id,
        department_path=row.department.path if row.department else None,
        is_active=row.is_active,
        must_change_password=row.must_change_password,
        has_mfa=row.mfa_secret is not None,
        permission_epoch=row.permission_epoch,
        failed_logins=row.failed_logins,
        locked_until=row.locked_until,
        last_login_at=row.last_login_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, user_id: UUID) -> User | None:
        row = await self._s.get(UserModel, user_id)
        if row is None or row.deleted_at is not None:
            return None
        return _to_user(row)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(UserModel).where(UserModel.email == email, UserModel.deleted_at.is_(None))
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return _to_user(row) if row else None

    async def get_credentials(self, user_id: UUID) -> UserCredentials | None:
        stmt = select(
            UserModel.id,
            UserModel.email,
            UserModel.password_hash,
            UserModel.mfa_secret,
            UserModel.recovery_code_hashes,
        ).where(UserModel.id == user_id, UserModel.deleted_at.is_(None))
        row = (await self._s.execute(stmt)).one_or_none()
        if row is None:
            return None
        return UserCredentials(
            id=row.id,
            email=row.email,
            password_hash=row.password_hash,
            mfa_secret=row.mfa_secret,
            recovery_code_hashes=list(row.recovery_code_hashes or []),
        )

    async def create(
        self,
        *,
        email: str,
        full_name: str,
        role: Role,
        password_hash: str | None,
        department_id: UUID | None = None,
        must_change_password: bool = False,
    ) -> User:
        row = UserModel(
            id=new_id(),
            email=email,
            full_name=full_name,
            role=role,
            password_hash=password_hash,
            department_id=department_id,
            must_change_password=must_change_password,
            permission_epoch=1,
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row, ["department"])
        return _to_user(row)

    async def update(self, user_id: UUID, **fields: Any) -> User:
        # Any change to role, department, or activity must invalidate issued tokens, so the
        # epoch bump lives here rather than in each caller that might forget it.
        if {"role", "department_id", "is_active"} & fields.keys():
            fields["permission_epoch"] = UserModel.permission_epoch + 1
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id, UserModel.deleted_at.is_(None))
            .values(**fields)
            .returning(UserModel.id)
        )
        if (await self._s.execute(stmt)).scalar_one_or_none() is None:
            raise NotFoundError("User", user_id)
        row = await self._s.get(UserModel, user_id, populate_existing=True)
        assert row is not None
        await self._s.refresh(row, ["department"])
        return _to_user(row)

    async def set_password(self, user_id: UUID, password_hash: str) -> None:
        await self._s.execute(
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(
                password_hash=password_hash,
                must_change_password=False,
                failed_logins=0,
                locked_until=None,
                permission_epoch=UserModel.permission_epoch + 1,
            )
        )

    async def set_mfa(
        self, user_id: UUID, *, secret: str | None, recovery_code_hashes: list[str] | None = None
    ) -> None:
        values: dict[str, Any] = {"mfa_secret": secret}
        if recovery_code_hashes is not None:
            values["recovery_code_hashes"] = recovery_code_hashes
        await self._s.execute(update(UserModel).where(UserModel.id == user_id).values(**values))

    async def consume_recovery_code(self, user_id: UUID, code_hash: str) -> bool:
        """Spend a recovery code in one statement.

        ``array_remove`` plus a ``WHERE`` on membership makes this atomic: two concurrent
        attempts with the same code cannot both succeed, which a read-then-write in Python
        could not guarantee.
        """
        stmt = (
            update(UserModel)
            .where(
                UserModel.id == user_id,
                UserModel.recovery_code_hashes.contains([code_hash]),
            )
            .values(
                recovery_code_hashes=func.array_remove(UserModel.recovery_code_hashes, code_hash)
            )
            .returning(UserModel.id)
        )
        return (await self._s.execute(stmt)).scalar_one_or_none() is not None

    async def record_login_success(self, user_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(last_login_at=at, failed_logins=0, locked_until=None)
        )

    async def record_login_failure(
        self, user_id: UUID, *, max_failures: int, lockout_minutes: int
    ) -> User:
        """Increment and conditionally lock, atomically.

        Expressed as one UPDATE with a CASE so that concurrent attempts cannot race past the
        threshold — the classic way brute-force protection is defeated in practice.
        """
        from sqlalchemy import case

        next_count = UserModel.failed_logins + 1
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(
                failed_logins=next_count,
                locked_until=case(
                    (
                        next_count >= max_failures,
                        func.now() + timedelta(minutes=lockout_minutes),
                    ),
                    else_=UserModel.locked_until,
                ),
            )
            .returning(UserModel.id)
        )
        if (await self._s.execute(stmt)).scalar_one_or_none() is None:
            raise NotFoundError("User", user_id)
        row = await self._s.get(UserModel, user_id, populate_existing=True)
        assert row is not None
        await self._s.refresh(row, ["department"])
        return _to_user(row)

    async def bump_permission_epoch(self, user_id: UUID) -> int:
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(permission_epoch=UserModel.permission_epoch + 1)
            .returning(UserModel.permission_epoch)
        )
        epoch = (await self._s.execute(stmt)).scalar_one_or_none()
        if epoch is None:
            raise NotFoundError("User", user_id)
        return int(epoch)

    async def list_users(
        self,
        *,
        role: Role | None = None,
        department_id: UUID | None = None,
        query: str | None = None,
        include_inactive: bool = False,
        limit: int = 50,
        cursor_created_at: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[User]:
        stmt = select(UserModel).where(UserModel.deleted_at.is_(None))
        if role is not None:
            stmt = stmt.where(UserModel.role == role)
        if department_id is not None:
            stmt = stmt.where(UserModel.department_id == department_id)
        if not include_inactive:
            stmt = stmt.where(UserModel.is_active.is_(True))
        if query:
            pattern = f"%{query}%"
            stmt = stmt.where(UserModel.full_name.ilike(pattern) | UserModel.email.ilike(pattern))
        if cursor_created_at is not None and cursor_id is not None:
            # Keyset: strictly older, with id as the tiebreaker for identical timestamps.
            stmt = stmt.where(
                (UserModel.created_at < cursor_created_at)
                | ((UserModel.created_at == cursor_created_at) & (UserModel.id < cursor_id))
            )
        stmt = stmt.order_by(UserModel.created_at.desc(), UserModel.id.desc()).limit(limit)
        rows = (await self._s.execute(stmt)).scalars().all()
        return [_to_user(r) for r in rows]

    async def count_by_role(self) -> dict[Role, int]:
        stmt = (
            select(UserModel.role, func.count())
            .where(UserModel.deleted_at.is_(None), UserModel.is_active.is_(True))
            .group_by(UserModel.role)
        )
        return {row[0]: int(row[1]) for row in (await self._s.execute(stmt)).all()}


class SqlDepartmentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, department_id: UUID) -> Department | None:
        row = await self._s.get(DepartmentModel, department_id)
        return Department.model_validate(row) if row else None

    async def get_by_slug(self, slug: str) -> Department | None:
        stmt = select(DepartmentModel).where(DepartmentModel.slug == slug)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return Department.model_validate(row) if row else None

    async def list_all(self) -> list[Department]:
        stmt = select(DepartmentModel).order_by(DepartmentModel.path)
        return [Department.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]

    async def create(self, *, name: str, slug: str, parent_id: UUID | None) -> Department:
        path = slug
        if parent_id is not None:
            parent = await self._s.get(DepartmentModel, parent_id)
            if parent is None:
                raise NotFoundError("Department", parent_id)
            path = f"{parent.path}.{slug}"
        row = DepartmentModel(id=new_id(), name=name, slug=slug, parent_id=parent_id, path=path)
        self._s.add(row)
        await self._s.flush()
        return Department.model_validate(row)

    async def subtree_paths(self, department_id: UUID) -> list[str]:
        root = await self._s.get(DepartmentModel, department_id)
        if root is None:
            return []
        stmt = select(DepartmentModel.path).where(DepartmentModel.path.op("<@")(ltree(root.path)))
        return [str(p) for p in (await self._s.execute(stmt)).scalars().all()]


class SqlRefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self,
        *,
        user_id: UUID,
        jti: UUID,
        token_hash: bytes,
        family_id: UUID,
        expires_at: datetime,
        user_agent: str | None,
        ip: str | None,
    ) -> None:
        self._s.add(
            RefreshTokenModel(
                id=new_id(),
                user_id=user_id,
                jti=jti,
                token_hash=token_hash,
                family_id=family_id,
                expires_at=expires_at,
                user_agent=user_agent[:500] if user_agent else None,
                ip=ip,
            )
        )
        await self._s.flush()

    async def find_active(self, token_hash: bytes) -> dict[str, Any] | None:
        stmt = select(
            RefreshTokenModel.user_id,
            RefreshTokenModel.jti,
            RefreshTokenModel.family_id,
            RefreshTokenModel.expires_at,
        ).where(
            RefreshTokenModel.token_hash == token_hash,
            RefreshTokenModel.revoked_at.is_(None),
            RefreshTokenModel.expires_at > func.now(),
        )
        row = (await self._s.execute(stmt)).one_or_none()
        if row is None:
            return None
        return {
            "user_id": row.user_id,
            "jti": row.jti,
            "family_id": row.family_id,
            "expires_at": row.expires_at,
        }

    async def was_used(self, token_hash: bytes) -> UUID | None:
        """Detect replay of a rotated token.

        A spent refresh token being presented means it leaked — either the client kept a copy
        or an attacker captured one — so the caller revokes the whole family rather than
        merely rejecting this request.
        """
        stmt = select(RefreshTokenModel.family_id).where(
            RefreshTokenModel.token_hash == token_hash,
            RefreshTokenModel.revoked_at.is_not(None),
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def revoke(self, jti: UUID) -> None:
        await self._s.execute(
            update(RefreshTokenModel)
            .where(RefreshTokenModel.jti == jti, RefreshTokenModel.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )

    async def revoke_family(self, family_id: UUID) -> int:
        result = await self._s.execute(
            update(RefreshTokenModel)
            .where(
                RefreshTokenModel.family_id == family_id,
                RefreshTokenModel.revoked_at.is_(None),
            )
            .values(revoked_at=func.now())
        )
        return affected(result)

    async def revoke_all_for_user(self, user_id: UUID, *, except_jti: UUID | None = None) -> int:
        stmt = update(RefreshTokenModel).where(
            RefreshTokenModel.user_id == user_id, RefreshTokenModel.revoked_at.is_(None)
        )
        if except_jti is not None:
            stmt = stmt.where(RefreshTokenModel.jti != except_jti)
        result = await self._s.execute(stmt.values(revoked_at=func.now()))
        return affected(result)

    async def purge_expired(self, *, before: datetime) -> int:
        result = await self._s.execute(
            delete(RefreshTokenModel).where(RefreshTokenModel.expires_at < before)
        )
        return affected(result)


class SqlApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self,
        *,
        name: str,
        prefix: str,
        key_hash: bytes,
        role: Role,
        scopes: list[str],
        created_by: UUID | None,
        expires_at: datetime | None,
    ) -> ApiKey:
        row = ApiKeyModel(
            id=new_id(),
            name=name,
            prefix=prefix,
            key_hash=key_hash,
            role=role,
            scopes=scopes,
            created_by=created_by,
            expires_at=expires_at,
        )
        self._s.add(row)
        await self._s.flush()
        return ApiKey.model_validate(row)

    async def find_by_hash(self, key_hash: bytes) -> ApiKey | None:
        stmt = select(ApiKeyModel).where(
            ApiKeyModel.key_hash == key_hash, ApiKeyModel.revoked_at.is_(None)
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return ApiKey.model_validate(row) if row else None

    async def touch(self, key_id: UUID, *, at: datetime) -> None:
        await self._s.execute(
            update(ApiKeyModel).where(ApiKeyModel.id == key_id).values(last_used_at=at)
        )

    async def revoke(self, key_id: UUID) -> None:
        await self._s.execute(
            update(ApiKeyModel)
            .where(ApiKeyModel.id == key_id, ApiKeyModel.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )

    async def list_all(self, *, include_revoked: bool = False) -> list[ApiKey]:
        stmt = select(ApiKeyModel).order_by(ApiKeyModel.created_at.desc())
        if not include_revoked:
            stmt = stmt.where(ApiKeyModel.revoked_at.is_(None))
        return [ApiKey.model_validate(r) for r in (await self._s.execute(stmt)).scalars().all()]


class SqlPermissionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def load_matrix(self) -> dict[Role, frozenset[Permission]]:
        stmt = select(RolePermissionModel.role, RolePermissionModel.permission_code)
        matrix: dict[Role, set[Permission]] = {}
        for role, code in (await self._s.execute(stmt)).all():
            try:
                matrix.setdefault(role, set()).add(Permission(code))
            except ValueError:
                # A code in the table that no longer exists in the enum: a removed capability
                # after a downgrade. Ignored rather than fatal — an unknown permission cannot
                # grant anything, since every check compares against an enum member.
                continue
        return {role: frozenset(perms) for role, perms in matrix.items()}

    async def replace_role_permissions(self, role: Role, permissions: set[Permission]) -> None:
        await self._s.execute(delete(RolePermissionModel).where(RolePermissionModel.role == role))
        if permissions:
            await self._s.execute(
                insert(RolePermissionModel),
                [{"role": role, "permission_code": p.value} for p in sorted(permissions)],
            )

    async def overrides_for(self, user_id: UUID) -> list[PermissionOverride]:
        stmt = select(
            UserPermissionOverrideModel.permission_code,
            UserPermissionOverrideModel.effect,
            UserPermissionOverrideModel.expires_at,
        ).where(UserPermissionOverrideModel.user_id == user_id)
        out: list[PermissionOverride] = []
        for code, effect, expires_at in (await self._s.execute(stmt)).all():
            try:
                out.append(
                    PermissionOverride(
                        permission=Permission(code), allow=effect == "allow", expires_at=expires_at
                    )
                )
            except ValueError:
                continue
        return out

    async def set_override(
        self,
        *,
        user_id: UUID,
        permission: Permission,
        allow: bool,
        granted_by: UUID | None,
        expires_at: datetime | None,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(UserPermissionOverrideModel).values(
            user_id=user_id,
            permission_code=permission.value,
            effect="allow" if allow else "deny",
            granted_by=granted_by,
            expires_at=expires_at,
        )
        await self._s.execute(
            stmt.on_conflict_do_update(
                index_elements=["user_id", "permission_code"],
                set_={
                    "effect": stmt.excluded.effect,
                    "granted_by": stmt.excluded.granted_by,
                    "expires_at": stmt.excluded.expires_at,
                },
            )
        )

    async def clear_override(self, *, user_id: UUID, permission: Permission) -> None:
        await self._s.execute(
            delete(UserPermissionOverrideModel).where(
                UserPermissionOverrideModel.user_id == user_id,
                UserPermissionOverrideModel.permission_code == permission.value,
            )
        )

    async def seed_defaults(self, matrix: dict[Role, frozenset[Permission]]) -> None:
        """Insert the permission catalogue and the default role matrix, idempotently."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        catalogue = [
            {"code": p.value, "category": category, "description": description}
            for p, (category, description) in PERMISSION_DESCRIPTIONS.items()
        ]
        if catalogue:
            stmt = pg_insert(PermissionModel).values(catalogue)
            await self._s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["code"],
                    set_={
                        "category": stmt.excluded.category,
                        "description": stmt.excluded.description,
                    },
                )
            )
        for role, permissions in matrix.items():
            await self.replace_role_permissions(role, set(permissions))
