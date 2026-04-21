"""Helpers for consistent CLI authentication guidance."""

from __future__ import annotations

from flow360.user_config import DEFAULT_PROFILE


def _env_flag(environment_name: str) -> str:
    if environment_name == "dev":
        return "--dev"
    if environment_name == "uat":
        return "--uat"
    if environment_name == "prod":
        return ""
    return f"--env {environment_name}"


def build_login_command(environment_name: str, profile: str) -> str:
    env_flag = _env_flag(environment_name)
    parts = ["flow360", "login"]
    if env_flag:
        parts.append(env_flag)
    if profile != DEFAULT_PROFILE:
        parts.extend(["--profile", profile])
    return " ".join(parts)


def build_configure_command(environment_name: str, profile: str) -> str:
    env_flag = _env_flag(environment_name)
    parts = ["flow360", "configure"]
    if env_flag:
        parts.append(env_flag)
    if profile != DEFAULT_PROFILE:
        parts.extend(["--profile", profile])
    parts.extend(["--apikey", "<apikey>"])
    return " ".join(parts)


def build_missing_api_key_message(environment_name: str, profile: str) -> str:
    return "\n".join(
        [
            f"No API key configured for env={environment_name}, profile={profile}.",
            "Authenticate with:",
            f"  {build_login_command(environment_name, profile)}",
            "For headless or manual setup:",
            f"  {build_configure_command(environment_name, profile)}",
        ]
    )
