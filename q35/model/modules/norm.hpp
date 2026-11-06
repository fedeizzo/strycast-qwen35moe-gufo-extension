#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_NORM_HPP_
#define GUFO_MODELS_QWEN_MODULES_NORM_HPP_

#include <span>

#include "q35/model/modules/layer_view.hpp"
#include "q35/model/modules/module_ctx.hpp"

namespace strycast::models::qwen {

/// Layer norm (attn pre-norm, ffn pre-norm, final output norm):
/// out = (x / rms(x) + eps) * weight.
///
/// CPU reproduces `ForwardRMSNorm`; HIP launches the existing RMSNorm kernel.
/// In-place is safe (`x == out`) on the CPU reference path.
void NormForward(const CpuModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;
void NormForward(const HipModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_NORM_HPP_
