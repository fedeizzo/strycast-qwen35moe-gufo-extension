#pragma once

#include "q35/ns.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "gufo/ext/text.hpp"
#include "q35/fnext/rope_scaling.hpp"

namespace strycast::models::qwen38_flash_next {

[[nodiscard]] std::size_t FlashNextBenchContext(
    const ext::TextBenchOptions& options);

[[nodiscard]] bool ResolveFlashNextBenchYarn(
    const ext::TextBenchOptions& options, std::uint32_t native,
    std::optional<RopeScaling>* scaling, std::string* error);

}  // namespace strycast::models::qwen38_flash_next

namespace strycast::cli {
using models::qwen38_flash_next::FlashNextBenchContext;
using models::qwen38_flash_next::ResolveFlashNextBenchYarn;
}  // namespace strycast::cli
