#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_UNEMBED_HPP_
#define GUFO_MODELS_QWEN_MODULES_UNEMBED_HPP_

#include <span>

#include "q35/model/modules/module_ctx.hpp"
#include "q35/model/state.hpp"

namespace strycast::models::qwen {

/// Final output: RMSNorm the hidden state then project through the LM head
/// onto logits. `output_weight` may share storage with the token embeddings
/// (tied LM head).
void UnembedForward(const CpuLayerContext& ctx,
                    const QwenTensorRef& output_norm,
                    const QwenTensorRef& output_weight,
                    std::span<const float> hidden,
                    std::span<float> logits_out) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_UNEMBED_HPP_
