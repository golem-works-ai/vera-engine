"""Tests for SecretRef, CredentialBundle, and the structural guard."""

from dataclasses import dataclass

import pytest

from vera_engine.credentials import (
    CredentialBundle,
    SecretRef,
    reject_credential_fields,
)


def test_secret_ref_holds_name():
    ref = SecretRef("ANTHROPIC_API_KEY")
    assert ref.name == "ANTHROPIC_API_KEY"


def test_secret_ref_empty_name_raises():
    with pytest.raises(ValueError, match="name must not be empty"):
        SecretRef("")


def test_secret_ref_repr_shows_name():
    ref = SecretRef("MY_TOKEN")
    assert repr(ref) == "SecretRef('MY_TOKEN')"


def test_secret_ref_str_does_not_look_like_a_value():
    ref = SecretRef("MY_TOKEN")
    text = str(ref)
    assert text == "<SecretRef:MY_TOKEN>"
    assert "MY_TOKEN" in text


def test_secret_ref_is_frozen():
    ref = SecretRef("X")
    with pytest.raises(Exception):
        ref.name = "Y"


def test_credential_bundle_resolve_success():
    bundle = CredentialBundle(values={"FOO": "secret-value"})
    assert bundle.resolve(SecretRef("FOO")) == "secret-value"


def test_credential_bundle_resolve_missing_raises_keyerror():
    bundle = CredentialBundle(values={"FOO": "bar"})
    with pytest.raises(KeyError, match="FOO2"):
        bundle.resolve(SecretRef("FOO2"))


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


def test_reject_credential_fields_allows_clean_dataclass():
    @dataclass
    class Clean:
        name: str
        workspace: str

    reject_credential_fields(Clean)  # should not raise


@pytest.mark.parametrize(
    "field_name",
    [
        "api_key",
        "api_secret",
        "token",
        "secret",
        "password",
        "credential",
        "private_key",
        "API_KEY",
        "my_api_key_value",
    ],
)
def test_reject_credential_fields_rejects_credential_shaped_names(field_name):
    ns = {"__annotations__": {field_name: str}}
    Dirty = dataclass(type("Dirty", (), ns))
    with pytest.raises(TypeError, match="looks like a credential value field"):
        reject_credential_fields(Dirty)


def test_reject_credential_fields_allows_credential_strategy_field():
    @dataclass
    class HasStrategy:
        credential_strategy: str

    reject_credential_fields(HasStrategy)  # explicitly allow-listed
