#include "q35/ns.hpp"
#ifndef GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <gufo/ext/experimental/managed_alloc.hpp>

namespace strycast::hip::detail {

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = strycast::hip::AllocateDevice(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP allocation failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

}  // namespace strycast::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
