#pragma once

// Strycast's copy of the Qwen code lives in namespace strycast, so it links
// beside Gufo's own Qwen extensions without duplicate symbols. Each strycast
// namespace that mirrors a Gufo one also sees Gufo's names, so core types
// such as hip::HipBuffer or ext::TextModelRunner still name Gufo's.

namespace gufo {}
namespace gufo::cache {}
namespace gufo::core { class GgufReader; }
namespace gufo::crypto {}
namespace gufo::ext {}
namespace gufo::hip {}
namespace gufo::hip::detail {}
namespace gufo::json {}
namespace gufo::platform {}
namespace gufo::quant {}
namespace gufo::sampling {}
namespace gufo::server {}
namespace gufo::speculative { struct SpeculativeOptions; }

namespace strycast { using namespace ::gufo; }
namespace strycast::cache { using namespace ::gufo::cache; }
namespace strycast::core { using namespace ::gufo::core; }
namespace strycast::crypto { using namespace ::gufo::crypto; }
namespace strycast::ext { using namespace ::gufo::ext; }
namespace strycast::hip { using namespace ::gufo::hip; }
namespace strycast::hip::detail { using namespace ::gufo::hip::detail; }
namespace strycast::json { using namespace ::gufo::json; }
namespace strycast::platform { using namespace ::gufo::platform; }
namespace strycast::quant { using namespace ::gufo::quant; }
namespace strycast::sampling { using namespace ::gufo::sampling; }
namespace strycast::server { using namespace ::gufo::server; }
namespace strycast::speculative { using namespace ::gufo::speculative; }
