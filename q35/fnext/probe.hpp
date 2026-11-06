#pragma once

#include "q35/ns.hpp"

#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace strycast::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);

}  // namespace strycast::models::qwen38_flash_next
