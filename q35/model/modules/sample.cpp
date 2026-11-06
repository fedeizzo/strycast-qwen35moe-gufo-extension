#include "q35/ns.hpp"
#include "q35/model/modules/sample.hpp"

#include "q35/model/forward.hpp"  // GreedyArgmax

namespace strycast::models::qwen {

std::uint32_t SampleForward(const CpuModuleContext&,
                            std::span<const float> logits) noexcept {
  // CPU backend: delegates to the existing greedy-argmax (already a small
  // standalone function). A sampling-policy module seam for the L1 tier.
  return GreedyArgmax(logits);
}

}  // namespace strycast::models::qwen
