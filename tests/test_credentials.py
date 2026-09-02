"""Tests for SecretRef, CredentialBundle, and the structural guard."""

from dataclasses import dataclass
from enum import StrEnum

import pytest

from vera_engine.credentials import (
    FORBIDDEN_FIELD_PATTERN,
    CredentialBundle,
    CredentialGuardError,
    SecretRef,
    assert_no_credential_shaped_fields,
    reject_credential_fields,
)

# ── SecretRef ────────────────────────────────────────────────────────────────


def test_secret_ref_holds_name():
    ref = SecretRef("ANTHROPIC_API_KEY")
    assert ref.name == "ANTHROPIC_API_KEY"


def test_secret_ref_empty_name_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        SecretRef("")


def test_secret_ref_whitespace_only_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        SecretRef("   ")


def test_secret_ref_repr_shows_name():
    ref = SecretRef("MY_TOKEN")
    assert repr(ref) == "SecretRef('MY_TOKEN')"


def test_secret_ref_str_does_not_look_like_a_value():
    ref = SecretRef("MY_TOKEN")
    text = str(ref)
    assert text == "<SecretRef:MY_TOKEN>"


def test_secret_ref_is_frozen():
    ref = SecretRef("X")
    with pytest.raises(Exception):
        ref.name = "Y"


def test_secret_ref_as_shell_reference():
    ref = SecretRef("PROXY_API_KEY")
    assert ref.as_shell_reference() == "$PROXY_API_KEY"


# ── CredentialBundle ─────────────────────────────────────────────────────────


def test_credential_bundle_resolve_success():
    bundle = CredentialBundle(values={"FOO": "secret-value"})
    assert bundle.resolve(SecretRef("FOO")) == "secret-value"


def test_credential_bundle_resolve_missing_raises_guard_error():
    bundle = CredentialBundle(values={"FOO": "bar"})
    with pytest.raises(CredentialGuardError, match="no value for 'FOO2'"):
        bundle.resolve(SecretRef("FOO2"))


def test_credential_bundle_resolve_missing_is_not_keyerror():
    bundle = CredentialBundle(values={})
    with pytest.raises(CredentialGuardError):
        bundle.resolve(SecretRef("MISSING"))
    try:
        bundle.resolve(SecretRef("MISSING"))
    except KeyError:
        pytest.fail("Should raise CredentialGuardError, not KeyError")
    except CredentialGuardError:
        pass


def test_credential_bundle_accepts_mapping():
    from collections.abc import Mapping

    mapping: Mapping[str, str] = {"A": "1", "B": "2"}
    bundle = CredentialBundle(values=mapping)
    assert bundle.resolve(SecretRef("A")) == "1"


def test_credential_bundle_repr_omits_values():
    bundle = CredentialBundle(values={"FOO": "supersecret"})
    text = repr(bundle)
    assert "supersecret" not in text
    assert "FOO" in text


def test_credential_bundle_str_omits_values():
    bundle = CredentialBundle(values={"FOO": "supersecret"})
    text = str(bundle)
    assert "supersecret" not in text


def test_credential_bundle_repr_sorted_keys():
    bundle = CredentialBundle(values={"ZKEY": "1", "AKEY": "2"})
    assert repr(bundle) == "CredentialBundle(keys=['AKEY', 'ZKEY'])"


# ── FORBIDDEN_FIELD_PATTERN ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "api_key",
        "api-key",
        "apikey",
        "API_KEY",
        "oauth",
        "secret",
        "credential",
        "password",
        "passwd",
        "my_api_key_value",
        "availability",
        "entitlement",
    ],
)
def test_forbidden_pattern_matches_credential_shapes(name):
    assert FORBIDDEN_FIELD_PATTERN.search(name), f"{name!r} should match"


@pytest.mark.parametrize(
    "name",
    [
        "model",
        "engine",
        "prompt",
        "workspace",
        "private_key",
    ],
)
def test_forbidden_pattern_ignores_safe_names(name):
    assert not FORBIDDEN_FIELD_PATTERN.search(name), f"{name!r} should not match"


# ── Structural guard ────────────────────────────────────────────────────────


def test_guard_allows_clean_dataclass():
    @dataclass
    class Clean:
        name: str
        workspace: str

    assert_no_credential_shaped_fields(Clean)


@pytest.mark.parametrize(
    "field_name",
    [
        "api_key",
        "oauth_token",
        "secret",
        "password",
        "credential",
        "passwd_hash",
        "API_KEY",
        "my_api_key_value",
    ],
)
def test_guard_rejects_credential_shaped_names(field_name):
    ns = {"__annotations__": {field_name: str}}
    Dirty = dataclass(type("Dirty", (), ns))
    with pytest.raises(CredentialGuardError):
        assert_no_credential_shaped_fields(Dirty)


def test_guard_allows_credential_strategy_str():
    @dataclass
    class HasStrategy:
        credential_strategy: str

    assert_no_credential_shaped_fields(HasStrategy)


def test_guard_allows_credential_strategy_strenum():
    class Strategy(StrEnum):
        PROXY = "proxy"
        NONE = "none"

    @dataclass
    class HasTypedStrategy:
        credential_strategy: Strategy

    assert_no_credential_shaped_fields(HasTypedStrategy)


def test_guard_extra_pattern():
    import re

    extra = re.compile(r"(bearer|cookie)", re.IGNORECASE)

    @dataclass
    class HasBearer:
        bearer_value: str

    with pytest.raises(CredentialGuardError):
        assert_no_credential_shaped_fields(HasBearer, extra_pattern=extra)

    @dataclass
    class NoCookie:
        name: str

    assert_no_credential_shaped_fields(NoCookie, extra_pattern=extra)


def test_reject_credential_fields_is_alias():
    assert reject_credential_fields is assert_no_credential_shaped_fields


# ── CredentialGuardError ─────────────────────────────────────────────────────


def test_credential_guard_error_is_runtime_error():
    assert issubclass(CredentialGuardError, RuntimeError)
