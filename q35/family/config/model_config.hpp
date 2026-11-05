#pragma once

#include "q35/ns.hpp"

#include <cstdint>

#include <gufo/ext/model_config.hpp>

namespace strycast::core {

/// Gufo's ModelConfig plus the routed-expert shape of qwen35moe targets. The
/// copy names this type as core::ModelConfig, so it replaces Gufo's within the
/// extension and still binds wherever the public API takes Gufo's.
struct ModelConfig : ::gufo::core::ModelConfig {
  // MoE (qwen35moe): zero expert_count denotes a dense FFN model.
  std::uint32_t expert_count{0};
  std::uint32_t expert_used_count{0};
  std::uint32_t expert_ff_length{0};
  std::uint32_t expert_shared_ff_length{0};

  [[nodiscard]] constexpr bool IsMoE() const noexcept {
    return expert_count > 0;
  }
};

}  // namespace strycast::core
