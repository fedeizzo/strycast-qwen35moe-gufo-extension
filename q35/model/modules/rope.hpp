#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_ROPE_HPP_
#define GUFO_MODELS_QWEN_MODULES_ROPE_HPP_

#include <cstdint>
#include <span>

#include "q35/model/modules/layer_view.hpp"
#include "q35/model/modules/module_ctx.hpp"

namespace strycast::models::qwen {

/// RoPE rotation on Q and K heads for the given position.
///
/// CPU backend reproduces the existing `ForwardRoPE` computation exactly
/// (`ReferenceRoPE` applied per head). In-place safe: `q`/`k` may be mutated
/// in place.
void RopeForward(const CpuModuleContext& ctx, const RopeLayerView& view,
                 std::span<float> q, std::span<float> k,
                 std::uint32_t pos) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_ROPE_HPP_
