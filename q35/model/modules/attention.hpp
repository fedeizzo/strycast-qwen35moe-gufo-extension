#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_ATTENTION_HPP_
#define GUFO_MODELS_QWEN_MODULES_ATTENTION_HPP_

#include <cstdint>
#include <span>

#include "q35/model/modules/layer_view.hpp"
#include "q35/model/modules/module_ctx.hpp"
#include "q35/model/state.hpp"

namespace strycast::models::qwen {

/// Full self-attention for one sequence position (gated GQA):
/// QKV proj -> QK-norm -> RoPE -> KV-cache write -> score/softmax/context ->
/// gate -> out proj.
///
/// CPU backend reproduces the exact attention branch formerly inlined in
/// `ForwardLayer` (both the fused `2*q_size` Q+gate projection and the plain
/// Q path), calling the kept `ForwardAttention` helper for the score/context
/// core. KV state lives in `kv`; required scratch/config capabilities are held
/// by the non-nullable CPU layer context.
void AttnForward(const CpuLayerContext& ctx, const AttnLayerView& view,
                 std::span<const float> x, QwenKvCache& kv, std::uint32_t pos,
                 std::span<float> out) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_ATTENTION_HPP_
