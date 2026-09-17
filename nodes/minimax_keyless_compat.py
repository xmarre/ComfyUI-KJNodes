from typing import Any

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"


class MiniMaxH3KeylessCompatibilityError(RuntimeError):
    """A MiniMax H3 patch cannot safely reinterpret an advertised Keyless model."""


def keyless_h3_contract(diffusion_model: Any) -> Any | None:
    """Return the validated public Keyless marker, or None for ordinary MiniMax H3.

    KJNodes deliberately does not import the Keyless package. This guard exists so
    QKV-specific patches fail before replacing a Keyless attention forward instead
    of discovering the missing K projection inside the sampler.
    """

    if diffusion_model is None or not hasattr(diffusion_model, KEYLESS_CONTRACT_KEY):
        return None

    contract = getattr(diffusion_model, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": 50,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": 2,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "projection_attr": "qv_proj",
        "qv_order": "q_effective;v",
    }
    mismatches = []
    for name, wanted in expected.items():
        if not hasattr(contract, name):
            mismatches.append(f"missing {name}")
            continue
        actual = getattr(contract, name)
        if actual != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise MiniMaxH3KeylessCompatibilityError(
            f"malformed {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches)
        )
    return contract


def reject_keyless_h3_qkv_patch(diffusion_model: Any, patch_name: str) -> None:
    """Reject an H3 patch whose implementation requires an ordinary QKV projection."""

    if keyless_h3_contract(diffusion_model) is None:
        return
    raise MiniMaxH3KeylessCompatibilityError(
        f"{patch_name} is QKV-only and cannot patch {KEYLESS_ARCHITECTURE}. "
        "Keyless main-block attention uses qv_proj and derives routing from V; "
        "the patch must not recreate or require a teacher K projection."
    )


def require_qkv_attention(attention: Any, patch_name: str) -> None:
    """Give stale/direct bindings a precise failure instead of an AttributeError."""

    if hasattr(attention, "qkv_proj"):
        return
    if hasattr(attention, "qv_proj"):
        raise MiniMaxH3KeylessCompatibilityError(
            f"{patch_name} requires qkv_proj but received Keyless qv_proj attention; "
            "refusing to synthesize or recover a K projection."
        )
    raise MiniMaxH3KeylessCompatibilityError(f"{patch_name} requires MiniMax H3 qkv_proj attention.")
