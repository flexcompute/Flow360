"""Shared CLI context helpers."""

from __future__ import annotations

from dataclasses import dataclass

from flow360.user_config import DEFAULT_PROFILE


@dataclass(frozen=True)
class CliContext:
    """Resolved CLI context shared across subcommands."""

    profile: str = DEFAULT_PROFILE
    dev: bool = False
    uat: bool = False
    env: str | None = None
    local: bool = False

    def as_dict(self) -> dict:
        return {
            "profile": self.profile,
            "dev": self.dev,
            "uat": self.uat,
            "env": self.env,
            "local": self.local,
        }


def resolve_root_context(profile=None, dev: bool = False, uat: bool = False, env=None) -> CliContext:
    """Resolve root CLI flags into a stable runtime context."""

    return CliContext(profile=profile or DEFAULT_PROFILE, dev=dev, uat=uat, env=env)


def merge_command_context(
    ctx,
    *,
    profile=None,
    dev: bool = False,
    uat: bool = False,
    env=None,
    local: bool = False,
) -> CliContext:
    """Merge command-local options with the already-resolved root context."""

    root_ctx = ctx.obj or {}
    return CliContext(
        profile=profile if profile is not None else root_ctx.get("profile", DEFAULT_PROFILE),
        dev=dev or root_ctx.get("dev", False),
        uat=uat or root_ctx.get("uat", False),
        env=env if env is not None else root_ctx.get("env"),
        local=local or root_ctx.get("local", False),
    )
