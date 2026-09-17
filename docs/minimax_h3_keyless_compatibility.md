# MiniMax H3 Keyless compatibility

KJNodes has two different kinds of attention integration for MiniMax H3, and they have different Keyless behavior.

The dedicated **MiniMax H3 Low VRAM Attention** and **MiniMax H3 Mem Eff Sage Attention Patch** implementations replace the native attention forward and explicitly read `qkv_proj`, `k_norm`, and the teacher K tensor. They are therefore QKV-only. When a diffusion model advertises `minimax_h3_keyless_contract_v1` / `h3_keyless_core50_v1`, these nodes fail before installing object patches. KJNodes does not create a compatibility `qkv_proj` alias, reconstruct K, or reinterpret V as K.

The generic KJ attention override is a different layer. It receives already-prepared Q/K/V tensors through ComfyUI's `optimized_attention_override` contract and does not inspect MiniMax weights. A canonical Keyless materialized-route path may therefore use the generic override as a dense compatibility backend after it has constructed the logical routing tensor from V. This does not make the dedicated QKV MiniMax patches Keyless-compatible, and it does not provide the native Keyless fused-memory result.

The MiniMax H3 feed-forward chunk patch and token counter do not replace attention projections and remain model-opaque.

KJNodes detects the Keyless architecture through its public runtime contract only. It deliberately does not import or depend on the Keyless custom-node package. An explicit but malformed Keyless contract fails closed instead of being treated as an ordinary QKV MiniMax model.
