#pragma once

#include "q35/ns.hpp"

#include <memory>
#include <string>

#include "gufo/ext/text.hpp"
#include "q35/model/dflash_policy.hpp"

namespace strycast::hip {
class QwenGpuModel;
class QwenDFlashGpuModel;
}  // namespace strycast::hip

namespace strycast::models::qwen {

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy dflash_policy,
    std::string artifact_fingerprint, std::string draft_artifact_fingerprint);
#endif

}  // namespace strycast::models::qwen
