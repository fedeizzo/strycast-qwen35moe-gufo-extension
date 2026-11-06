#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>

#include "q35/family/kernels/ops/ssm.hpp"
#include "q35/family/text/tokenizer.hpp"
#include "q35/family/vision/device_input.hpp"

namespace strycast::models {
struct QwenModelWeights;
}

namespace strycast::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Emits pure decode route decisions before entering graph capture. This must
/// never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy);

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
/// `recurrent_state` is the pass's QwenGpuArena::BeginRecurrentWrite result;
/// a captured graph bakes its pointers, so the graph key must carry
/// QwenGpuArena::RecurrentIoIdentity.
void ExecuteDecodeStep(
    QwenGpuArena& arena, const models::QwenModelWeights& weights,
    const QwenExecutionPolicy& policy, tokenization::TokenId token_id,
    std::uint32_t pos, bool compute_logits,
    QwenRecurrentStateIo recurrent_state,
    models::qwen::vision::DeviceInput* image_input = nullptr);

}  // namespace strycast::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
