#include "q35/ns.hpp"
#include "q35/model/modules/residual.hpp"

#include <cstddef>

#if defined(ENGINE_ENABLE_HIP)
#include "q35/family/kernels/ops/norm_residual.hpp"
#endif

namespace strycast::models::qwen {

void ResidualAdd(const CpuModuleContext&, std::span<float> dst,
                 std::span<const float> src) noexcept {
  for (std::size_t i = 0; i < dst.size(); ++i) {
    dst[i] += src[i];
  }
}

#if defined(ENGINE_ENABLE_HIP)
void ResidualAdd(const HipModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept {
  ::strycast::hip::LaunchResidualAdd(dst.data(), src.data(), dst.data(), dst.size(),
                                 static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace strycast::models::qwen
