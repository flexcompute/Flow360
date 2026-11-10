"""Shared CLI context helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from flow360.user_config import DEFAULT_PROFILE


def activate_profile(ctx, profile: str) -> None:
    """Switch the API key profile for this command and restore it on close."""
    # pylint: disable=import-outside-toplevel
    from flow360 import user_config

    prev_profile_env = os.environ.get("SIMCLOUD_PROFILE")
    prev_profile = user_config.UserConfig.profile
    os.environ["SIMCLOUD_PROFILE"] = profile
    user_config.UserConfig.set_profile(profile)

    def restore_profile():
        if prev_profile_env is None:
            os.environ.pop("SIMCLOUD_PROFILE", None)
        else:
            os.environ["SIMCLOUD_PROFILE"] = prev_profile_env
        user_config.UserConfig.set_profile(prev_profile)

    ctx.call_on_close(restore_profile)


@dataclass(frozen=True)
class CliContext:
    """Resolved CLI context shared across subcommands."""

    profile: str = DEFAULT_PROFILE
    dev: bool = False
    uat: bool = False
    env: str | None = None

    def as_dict(self) -> dict:
        """Return a Click context-compatible dictionary."""

        return {
            "profile": self.profile,
            "dev": self.dev,
            "uat": self.uat,
            "env": self.env,
        }


def resolve_root_context(
    profile=None, dev: bool = False, uat: bool = False, env=None
) -> CliContext:
    """Resolve root CLI flags into a stable runtime context."""

    return CliContext(profile=profile or DEFAULT_PROFILE, dev=dev, uat=uat, env=env)


def merge_command_context(
    ctx,
    *,
    profile=None,
    dev: bool = False,
    uat: bool = False,
    env=None,
) -> CliContext:
    """Merge command-local options with the already-resolved root context."""

    root_ctx = ctx.obj or {}
    return CliContext(
        profile=profile if profile is not None else root_ctx.get("profile", DEFAULT_PROFILE),
        dev=dev or root_ctx.get("dev", False),
        uat=uat or root_ctx.get("uat", False),
        env=env if env is not None else root_ctx.get("env"),
    )
