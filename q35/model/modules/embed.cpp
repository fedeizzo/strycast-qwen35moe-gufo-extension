#include "q35/ns.hpp"
#include "q35/model/modules/embed.hpp"

#include "q35/model/forward.hpp"  // ForwardEmbedding

namespace strycast::models::qwen {

void EmbedForward(const CpuModuleContext&, std::uint32_t token_id,
                  const QwenTensorRef& token_embd, std::size_t hidden_size,
                  std::span<float> out) noexcept {
  // CPU backend: delegates to the existing embedding lookup (already a small
  // standalone function; no body duplicated).
  ForwardEmbedding(token_id, token_embd, hidden_size, out);
}

}  // namespace strycast::models::qwen
