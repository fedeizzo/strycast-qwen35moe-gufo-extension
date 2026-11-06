#include "q35/ns.hpp"
#include "q35/model/modules/quant_gemm.hpp"

#include <cstdlib>

#include "q35/family/hip_common/gemm_route.hpp"
#include "q35/model/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "q35/family/kernels/ops/gemm.hpp"
#endif

namespace strycast::models::qwen {

void QuantGemm(const CpuModuleContext&, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  TensorGEMV(A, x, M, K, y);
}

#if defined(ENGINE_ENABLE_HIP)
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  const auto resolution =
      ResolveQwenGemmRoute({.type = A.type,
                            .batch_size = 1,
                            .m = M,
                            .k = K,
                            .mode = QwenGemmMode::kHipDecode});
  if (A.empty() || !resolution.accepted() || x.size() < K || y.size() < M ||
      A.num_elements < M * K) {
    std::abort();
  }
  ::strycast::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                          static_cast<hipStream_t>(ctx.Stream()));
}
#endif

}  // namespace strycast::models::qwen
