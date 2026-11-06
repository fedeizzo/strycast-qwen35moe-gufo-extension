#include "q35/ns.hpp"
#include "q35/model/modules/norm.hpp"

#include <cstddef>
#include <vector>

#include "q35/model/oracles.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "q35/family/kernels/ops/norm_residual.hpp"
#endif

namespace strycast::models::qwen {

void NormForward(const CpuModuleContext&, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept {
  if (view.weight.type == core::GgmlType::kF32) {
    ReferenceRMSNorm(x, view.weight.AsFloatSpan(), view.eps, out);
  } else {
    std::vector<float> w_f32(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      w_f32[i] = view.weight.Get(i);
    }
    ReferenceRMSNorm(x, w_f32, view.eps, out);
  }
}

#if defined(ENGINE_ENABLE_HIP)
void NormForward(const HipModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept {
  ::strycast::hip::LaunchRMSNorm(
      x.data(), static_cast<const float*>(view.weight.data), out.data(),
      x.size(), view.eps, static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace strycast::models::qwen
