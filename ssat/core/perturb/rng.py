"""Item-local random seed derivation."""

from __future__ import annotations

import hashlib

_RNG_NAMESPACE = b"ssat:rng:v1"


def derive(global_seed: int, item_id: str, seed_salt: int) -> int:
    """Derive a stable NumPy seed from one work item's identity.

    Args:
        global_seed: Non-negative seed shared by the audit run.
        item_id: Lowercase SHA-256 work item identifier.
        seed_salt: Non-negative perturbation variant salt.

    Returns:
        A deterministic unsigned 128-bit integer for ``default_rng``.

    Raises:
        ValueError: If a seed, salt, or item identifier is invalid.
    """

    if isinstance(global_seed, bool) or not isinstance(global_seed, int) or global_seed < 0:
        raise ValueError("global_seed must be a non-negative integer")
    if isinstance(seed_salt, bool) or not isinstance(seed_salt, int) or seed_salt < 0:
        raise ValueError("seed_salt must be a non-negative integer")
    if (
        not isinstance(item_id, str)
        or len(item_id) != 64
        or any(character not in "0123456789abcdef" for character in item_id)
    ):
        raise ValueError(
            "item_id must be a 64-character lowercase SHA-256 hex digest"
        )

    payload = b"\0".join(
        (
            _RNG_NAMESPACE,
            str(global_seed).encode("ascii"),
            item_id.encode("ascii"),
            str(seed_salt).encode("ascii"),
        )
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)
