#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_MODULES_RESIDUAL_HPP_
#define GUFO_MODELS_QWEN_MODULES_RESIDUAL_HPP_

#include <span>

#include "q35/model/modules/module_ctx.hpp"

namespace strycast::models::qwen {

/// Residual add: dst = dst + src (element-wise). In-place on `dst`.
///
/// The fused residual+norm route is a composition-layer concern (owned by the
/// per-layer `ExecuteStep`), not a module-local one, so the module is the
/// standalone add only.
void ResidualAdd(const CpuModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;
void ResidualAdd(const HipModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;

}  // namespace strycast::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_RESIDUAL_HPP_
