"""Credential isolation primitives.

SecretRef carries a name, never a value. CredentialBundle exists only at
spawn time. Structural guards prevent credential-shaped fields from
appearing on data objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

# Patterns that indicate a field is carrying a credential value.
_CREDENTIAL_PATTERNS = re.compile(
    r"(api_key|api_secret|token|secret|password|credential|private_key)",
    re.IGNORECASE,
)

# Fields that are allowed despite matching the pattern.
_ALLOWED_FIELDS = frozenset({"credential_strategy"})


@dataclass(frozen=True)
class SecretRef:
    """A reference to a secret BY NAME. Carries no value.

    The actual value is resolved at spawn time from a CredentialBundle.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SecretRef name must not be empty")

    def __repr__(self) -> str:
        return f"SecretRef({self.name!r})"

    # Prevent accidental str() from exposing the name in logs as if it were a value.
    def __str__(self) -> str:
        return f"<SecretRef:{self.name}>"


@dataclass(frozen=True)
class CredentialBundle:
    """Spawn-time secret values. Never serialized, never logged.

    Lives only long enough to resolve SecretRefs into env vars for
    subprocess.run().
    """

    values: dict[str, str]

    def resolve(self, ref: SecretRef) -> str:
        """Resolve a SecretRef to its value. Raises if missing."""
        try:
            return self.values[ref.name]
        except KeyError:
            raise KeyError(
                f"credential {ref.name!r} not found in bundle "
                f"(available: {sorted(self.values.keys())})"
            ) from None

    def __repr__(self) -> str:
        names = sorted(self.values.keys())
        return f"CredentialBundle(keys={names})"

    # Never expose values in str().
    def __str__(self) -> str:
        return self.__repr__()


def reject_credential_fields(cls: type) -> None:
    """Raise if a dataclass has fields whose names look like credential values.

    Call this on request/invocation types to enforce the SecretRef discipline.
    """
    for f in fields(cls):
        if f.name in _ALLOWED_FIELDS:
            continue
        if _CREDENTIAL_PATTERNS.search(f.name):
            raise TypeError(
                f"{cls.__name__}.{f.name} looks like a credential value field. "
                f"Use SecretRef for credential references, "
                f"CredentialBundle for spawn-time values."
            )
