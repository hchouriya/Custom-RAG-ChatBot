"""Identity, roles, permissions, sessions, and service credentials."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aegis.domain.enums import Role
from aegis.infrastructure.database.models.base import (
    CITEXT,
    LTREE,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    fk_uuid,
    pg_enum,
    uuid_pk,
)


class DepartmentModel(Base):
    __tablename__ = "departments"
    # GiST on the ltree path: every retrieval request for a manager or above evaluates a
    # subtree-containment operator against it.
    __table_args__ = (Index("ix_departments_path", "path", postgresql_using="gist"),)

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    parent_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    path: Mapped[str] = mapped_column(LTREE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )

    children: Mapped[list[DepartmentModel]] = relationship(
        back_populates="parent", cascade="all", lazy="noload"
    )
    parent: Mapped[DepartmentModel | None] = relationship(
        back_populates="children", remote_side=[id], lazy="noload"
    )


class UserModel(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (
        Index(
            "ix_users_role_active",
            "role",
            postgresql_where=text("deleted_at IS NULL AND is_active"),
        ),
        Index("ix_users_department", "department_id", postgresql_where=text("deleted_at IS NULL")),
    )

    id: Mapped[UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    # Nullable: SSO-only principals and guest rows have no password.
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[Role] = mapped_column(pg_enum(Role, "user_role"), nullable=False)
    department_id: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    mfa_secret: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovery_code_hashes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    external_idp_sub: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    # Bumped on any change that must invalidate issued access tokens. See core.security.
    permission_epoch: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    failed_logins: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    department: Mapped[DepartmentModel | None] = relationship(lazy="joined")


class PermissionModel(Base):
    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)


class RolePermissionModel(Base):
    __tablename__ = "role_permissions"

    role: Mapped[Role] = mapped_column(pg_enum(Role, "user_role"), primary_key=True)
    permission_code: Mapped[str] = mapped_column(
        Text, ForeignKey("permissions.code", ondelete="CASCADE"), primary_key=True
    )


class UserPermissionOverrideModel(Base, CreatedAtMixin):
    """Per-user exception to the role matrix.

    Exists because in every real deployment someone needs one capability above their role,
    and the alternative is inventing a fake role per exception.
    """

    __tablename__ = "user_permission_overrides"
    __table_args__ = (CheckConstraint("effect IN ('allow','deny')", name="effect_valid"),)

    user_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    permission_code: Mapped[str] = mapped_column(
        Text, ForeignKey("permissions.code", ondelete="CASCADE"), primary_key=True
    )
    effect: Mapped[str] = mapped_column(String(5), nullable=False)
    granted_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RefreshTokenModel(Base, CreatedAtMixin):
    """Rotating refresh tokens with family-based reuse detection.

    Only the digest is stored, so a database compromise yields no usable session. Presenting
    a token whose ``jti`` was already rotated away means it leaked, and the response is to
    revoke the entire ``family_id`` rather than only that token.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("ix_refresh_active", "user_id", postgresql_where=text("revoked_at IS NULL")),
        Index("ix_refresh_family", "family_id"),
        Index("ix_refresh_token_hash", "token_hash", unique=True),
    )

    id: Mapped[UUID] = uuid_pk()
    user_id: Mapped[UUID] = fk_uuid("users.id")
    jti: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False, unique=True)
    token_hash: Mapped[bytes] = mapped_column(nullable=False)
    family_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(INET, nullable=True)


class ApiKeyModel(Base, CreatedAtMixin):
    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("prefix", name="uq_api_keys_prefix"),
        # Authentication is a lookup by digest, so it must be indexed; unique because two
        # keys hashing alike would make the winner arbitrary.
        Index("ix_api_keys_key_hash", "key_hash", unique=True),
    )

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    key_hash: Mapped[bytes] = mapped_column(nullable=False)
    role: Mapped[Role] = mapped_column(pg_enum(Role, "user_role"), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    created_by: Mapped[UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


__all__ = [
    "ApiKeyModel",
    "DepartmentModel",
    "PermissionModel",
    "RefreshTokenModel",
    "RolePermissionModel",
    "UserModel",
    "UserPermissionOverrideModel",
]
