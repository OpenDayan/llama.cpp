// Included with MEGA_GRAPH_BEGIN, MEGA_STEP, and MEGA_GRAPH_END defined.
#define MEGA_FFN_FUSED
#define MEGA_MXFP4
MEGA_GRAPH_BEGIN(Qwen35DenseMxfp4LinearFfn, kDayanMegaQwen35DenseMxfp4LinearFfn, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35DenseMxfp4FullFfn, kDayanMegaQwen35DenseMxfp4FullFfn, false, false)
#include "full.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
#undef MEGA_MXFP4

MEGA_GRAPH_BEGIN(Qwen35DenseBf16LinearFfn, kDayanMegaQwen35DenseBf16LinearFfn, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35DenseBf16FullFfn, kDayanMegaQwen35DenseBf16FullFfn, false, false)
#include "full.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35MoeLinearFfn, kDayanMegaQwen35MoeLinearFfn, true, true)
#include "linear.inc"
#include "moe-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35MoeFullFfn, kDayanMegaQwen35MoeFullFfn, true, false)
#include "full.inc"
#include "moe-ffn.inc"
MEGA_GRAPH_END()
#undef MEGA_FFN_FUSED

#define MEGA_MXFP4
MEGA_GRAPH_BEGIN(Qwen35DenseMxfp4Linear, kDayanMegaQwen35DenseMxfp4Linear, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35DenseMxfp4Full, kDayanMegaQwen35DenseMxfp4Full, false, false)
#include "full.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
#undef MEGA_MXFP4

MEGA_GRAPH_BEGIN(Qwen35DenseBf16Linear, kDayanMegaQwen35DenseBf16Linear, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35DenseBf16Full, kDayanMegaQwen35DenseBf16Full, false, false)
#include "full.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35MoeLinear, kDayanMegaQwen35MoeLinear, true, true)
#include "linear.inc"
#include "moe-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35MoeFull, kDayanMegaQwen35MoeFull, true, false)
#include "full.inc"
#include "moe-ffn.inc"
MEGA_GRAPH_END()
