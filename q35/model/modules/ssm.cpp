#include "q35/ns.hpp"
#include "q35/model/modules/modules.hpp"

namespace strycast::models::qwen {

void SsmForward(const CpuLayerContext& ctx, const SsmLayerView& view,
                std::span<const float> x, QwenSsmCache& state,
                std::span<float> out) noexcept {
  auto& scratch = ctx.Scratch();
  ::strycast::models::ForwardSSM(x, view, state, ctx.LayerIndex(), scratch.ssm_qkv,
                             scratch.ssm_gate, scratch.ssm_out_buf, out);
}

}  // namespace strycast::models::qwen
