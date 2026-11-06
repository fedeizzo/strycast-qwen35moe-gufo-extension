#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
#define GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_

#include <cstdint>
#include <span>

#include "q35/model/modules/module_ctx.hpp"

namespace strycast::models::qwen {

/// Greedy argmax over a logit distribution (sampling-policy module seam).
std::uint32_t SampleForward(const CpuModuleContext& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
