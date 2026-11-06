#include "q35/ns.hpp"
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "q35/model/probe.hpp"

#if !defined(ENGINE_ENABLE_HIP)
#include <chrono>
#include <iostream>

#include "q35/model/generator.hpp"

namespace strycast::models::qwen {
int RunQwenPromptCpu(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  std::string err;
  auto generator = models::QwenGenerator::CreateFromGguf(reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    return 1;
  }
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    model_load_start)
          .count();
  std::cout << "[Model Load]: " << load_seconds << " s\n";

  auto prompt_tokens = generator->GetTokenizer().Encode(opt.prompt_text);
  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.sampling = opt.sampling;

  const auto start = std::chrono::steady_clock::now();
  auto generated = generator->Generate(
      prompt_tokens, gen_opts,
      [](tokenization::TokenId, std::string_view token_str) {
        std::cout << token_str << std::flush;
        return true;
      });
  const double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cout << "\n\nGenerated " << generated.size() << " tokens on CPU in "
            << sec << "s (" << (sec > 0 ? generated.size() / sec : 0)
            << " tok/s)\n";
  return 0;
}
}  // namespace strycast::models::qwen

void strycast_qwen35moe_register(strycast::ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "qwen35moe",
      .probe = &strycast::models::qwen::ProbeQwen,
      .probe_gguf = &strycast::models::qwen::ProbeQwenGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = &strycast::models::qwen::RunQwenPromptCpu,
      .chat_driver = nullptr,
      .sampling_defaults = &strycast::models::qwen::QwenSamplingDefaults,
  });
}
#else
namespace strycast::models::qwen {}  // namespace strycast::models::qwen

void strycast_qwen35moe_register(strycast::ext::Registry&) {}
#endif
