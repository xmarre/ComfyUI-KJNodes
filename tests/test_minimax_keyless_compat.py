from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "nodes" / "minimax_keyless_compat.py"

spec = importlib.util.spec_from_file_location("minimax_keyless_compat_test_target", HELPER)
compat = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compat)


def keyless_contract(**changes):
    values = {
        "api": 1,
        "architecture": "h3_keyless_core50_v1",
        "core_blocks": 50,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": 2,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "projection_attr": "qv_proj",
        "qv_order": "q_effective;v",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def keyless_model(**changes):
    return SimpleNamespace(minimax_h3_keyless_contract_v1=keyless_contract(**changes))


class KeylessGuardTests(unittest.TestCase):
    def test_ordinary_model_is_not_classified_as_keyless(self):
        self.assertIsNone(compat.keyless_h3_contract(SimpleNamespace()))

    def test_canonical_public_contract_is_recognized(self):
        model = keyless_model()
        self.assertIs(
            compat.keyless_h3_contract(model),
            model.minimax_h3_keyless_contract_v1,
        )

    def test_malformed_explicit_contract_fails_closed(self):
        with self.assertRaisesRegex(
            compat.MiniMaxH3KeylessCompatibilityError,
            "malformed minimax_h3_keyless_contract_v1",
        ):
            compat.keyless_h3_contract(keyless_model(routing_source="key"))

    def test_qkv_patch_rejects_keyless_before_forward_replacement(self):
        with self.assertRaisesRegex(
            compat.MiniMaxH3KeylessCompatibilityError,
            "QKV-only",
        ):
            compat.reject_keyless_h3_qkv_patch(keyless_model(), "test-patch")

    def test_direct_binding_rejects_qv_attention_without_synthesizing_k(self):
        with self.assertRaisesRegex(
            compat.MiniMaxH3KeylessCompatibilityError,
            "refusing to synthesize or recover a K projection",
        ):
            compat.require_qkv_attention(SimpleNamespace(qv_proj=object()), "test-patch")

    def test_native_qkv_attention_is_unchanged(self):
        compat.require_qkv_attention(SimpleNamespace(qkv_proj=object()), "test-patch")


class SourceIntegrationTests(unittest.TestCase):
    def test_ffn_chunk_patch_remains_keyless_model_opaque(self):
        source = (ROOT / "nodes" / "minimax_nodes.py").read_text(encoding="utf-8")
        start = source.index("class MiniMaxChunkFeedForward")
        end = source.index("def minimax_attn_lowmem_forward", start)
        chunker = source[start:end]
        self.assertNotIn("reject_keyless_h3_qkv_patch", chunker)
        self.assertNotIn("qkv_proj", chunker)

    def test_low_vram_patch_guards_before_qkv_model_shape_probe(self):
        source = (ROOT / "nodes" / "minimax_nodes.py").read_text(encoding="utf-8")
        execute = source.index("class MiniMaxLowVRAMAttention")
        guard = source.index(
            'reject_keyless_h3_qkv_patch(diffusion_model, "MiniMaxLowVRAMAttention")',
            execute,
        )
        qkv_probe = source.index('hasattr(blocks[0].attn, "qkv_proj")', execute)
        self.assertLess(guard, qkv_probe)

        forward = source.index("def minimax_attn_lowmem_forward")
        direct_guard = source.index(
            'require_qkv_attention(self, "MiniMaxLowVRAMAttention")',
            forward,
        )
        qkv_use = source.index("self.qkv_proj(x)", forward)
        self.assertLess(direct_guard, qkv_use)

    def test_minimax_mem_eff_sage_guards_before_direct_qkv_forward(self):
        source = (ROOT / "nodes" / "ltxv_nodes.py").read_text(encoding="utf-8")
        node = source.index("class MiniMaxH3MemoryEfficientSageAttentionPatch")
        guard = source.index(
            'reject_keyless_h3_qkv_patch(diffusion_model, "MiniMaxH3MemoryEfficientSageAttentionPatch")',
            node,
        )
        patch = source.index('add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward"', node)
        self.assertLess(guard, patch)

        forward = source.index("def minimax_sageattn_forward")
        direct_guard = source.index(
            'require_qkv_attention(self, "MiniMaxH3MemoryEfficientSageAttentionPatch")',
            forward,
        )
        qkv_use = source.index("self.qkv_proj(x)", forward)
        self.assertLess(direct_guard, qkv_use)

    def test_generic_attention_override_remains_model_agnostic(self):
        source = (ROOT / "nodes" / "model_optimization_nodes.py").read_text(encoding="utf-8")
        self.assertIn('"optimized_attention_override"', source)
        self.assertNotIn("qkv_proj", source)
        self.assertNotIn("minimax_keyless_compat", source)


if __name__ == "__main__":
    unittest.main()
