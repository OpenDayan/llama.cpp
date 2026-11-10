/*
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
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

#define MEGA_GDN_Z_GATE
MEGA_GRAPH_BEGIN(Qwen35DenseBf16LinearGdnFfn, kDayanMegaQwen35DenseBf16LinearGdnFfn, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
#undef MEGA_GDN_Z_GATE
MEGA_GRAPH_BEGIN(Qwen35DenseBf16LinearFfn, kDayanMegaQwen35DenseBf16LinearFfn, false, true)
#include "linear.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
MEGA_GRAPH_BEGIN(Qwen35DenseBf16FullFfn, kDayanMegaQwen35DenseBf16FullFfn, false, false)
#include "full.inc"
#include "dense-ffn.inc"
MEGA_GRAPH_END()
#define MEGA_GDN_Z_GATE
MEGA_GRAPH_BEGIN(Qwen35MoeLinearGdnFfn, kDayanMegaQwen35MoeLinearGdnFfn, true, true)
#include "linear.inc"
#include "moe-ffn.inc"
MEGA_GRAPH_END()
#undef MEGA_GDN_Z_GATE
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
