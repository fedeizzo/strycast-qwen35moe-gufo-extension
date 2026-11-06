#ifndef GUFO_MODELS_QWEN_HIP_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "q35/family/kernels/ops/attention.hpp"
#include "q35/family/kernels/ops/gemm.hpp"
#include "q35/family/kernels/ops/norm_residual.hpp"
#include "q35/family/kernels/ops/ssm.hpp"
#include "q35/family/kernels/ops/swiglu.hpp"
#include "q35/family/kernels/ops/token.hpp"

#endif  // GUFO_MODELS_QWEN_HIP_OPS_HPP_
