"""Credential isolation primitives.

SecretRef carries a name, never a value. CredentialBundle exists only at
spawn time. Structural guards prevent credential-shaped fields from
appearing on data objects.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import get_args, get_type_hints

FORBIDDEN_FIELD_PATTERN = re.compile(
    r"(api[_-]?key|oauth|secret|credential|password|passwd|availab|entitle)",
    re.IGNORECASE,
)


class CredentialGuardError(RuntimeError):
    """A structural credential rule would have been violated."""


@dataclass(frozen=True)
class SecretRef:
    """A reference to a secret BY NAME. Carries no value.

    The actual value is resolved at spawn time from a CredentialBundle.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("SecretRef name must not be empty")

    def as_shell_reference(self) -> str:
        """The bash spelling of this reference."""
        return f"${self.name}"

    def __repr__(self) -> str:
        return f"SecretRef({self.name!r})"

    def __str__(self) -> str:
        return f"<SecretRef:{self.name}>"


@dataclass(frozen=True)
class CredentialBundle:
    """Spawn-time secret values. Never serialized, never logged.

    Lives only long enough to resolve SecretRefs into env vars for
    subprocess.run().
    """

    values: Mapping[str, str]

    def resolve(self, ref: SecretRef) -> str:
        """Resolve a SecretRef to its value. Raises CredentialGuardError if missing."""
        try:
            return self.values[ref.name]
        except KeyError:
            msg = (
                f"CredentialBundle has no value for {ref.name!r}. "
                "Fail fast: a missing secret is a configuration error, not an "
                "empty string."
            )
            raise CredentialGuardError(msg) from None

    def __repr__(self) -> str:
        names = sorted(self.values.keys())
        return f"CredentialBundle(keys={names})"

    def __str__(self) -> str:
        return self.__repr__()


def _is_strategy_label(cls: type, name: str) -> bool:
    """Whether the field is a strategy LABEL rather than a credential value.

    Exempts ``credential_strategy`` only when annotated with a closed set
    (a StrEnum or similar), not when typed as bare ``str``.
    """
    if name != "credential_strategy":
        return False
    try:
        hints = get_type_hints(cls)
    except Exception:
        return False
    hint = hints.get(name)
    if hint is None:
        return False
    # Accept str for vera-engine's own request.py (where the validator
    # constrains to _VALID_STRATEGIES at runtime). Also accept StrEnum
    # subclasses and Optional[StrEnum] for Vera's typed strategy.
    members = set(get_args(hint)) or {hint}
    if str in members:
        return True
    from enum import StrEnum

    return all(
        m is type(None) or (isinstance(m, type) and issubclass(m, StrEnum))
        for m in members
    )


def assert_no_credential_shaped_fields(
    cls: type,
    *,
    extra_pattern: re.Pattern[str] | None = None,
) -> None:
    """Raise when ``cls`` declares a field that could hold a credential.

    Uses FORBIDDEN_FIELD_PATTERN. Callers can pass ``extra_pattern`` to
    widen the check for their package without forking the base pattern.
    """
    leaky: list[str] = []
    for f in fields(cls):
        if _is_strategy_label(cls, f.name):
            continue
        if (
            FORBIDDEN_FIELD_PATTERN.search(f.name)
            or extra_pattern
            and extra_pattern.search(f.name)
        ):
            leaky.append(f.name)
    if leaky:
        msg = (
            f"{cls.__name__} declares credential-shaped field(s) {leaky!r}. "
            "Use SecretRef for credential references, "
            "CredentialBundle for spawn-time values."
        )
        raise CredentialGuardError(msg)


# Keep the old name as an alias for backward compatibility with callers
# that imported `reject_credential_fields`.
reject_credential_fields = assert_no_credential_shaped_fields
