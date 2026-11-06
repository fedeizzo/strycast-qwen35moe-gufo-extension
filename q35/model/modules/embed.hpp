#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_EMBED_HPP_
#define GUFO_MODELS_QWEN_MODULES_EMBED_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "q35/model/modules/module_ctx.hpp"
#include "q35/model/state.hpp"

namespace strycast::models::qwen {

/// Token embedding lookup: copies the embedding row for token_id into out.
void EmbedForward(const CpuModuleContext& ctx, std::uint32_t token_id,
                  const QwenTensorRef& token_embd, std::size_t hidden_size,
                  std::span<float> out) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_EMBED_HPP_
