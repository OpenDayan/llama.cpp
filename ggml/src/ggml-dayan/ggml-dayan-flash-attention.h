#pragma once

#include "ggml-dayan-activation.h"
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#include <stdint.h>

namespace ggml_dayan_flash_attention_detail {

constexpr long kSlices = 2;
constexpr long kSlots = 2;
constexpr long kVecLanes = 64;
constexpr uint32_t kFlagCausal = 1;
constexpr long kBr = 128;
constexpr long kBc = 128;
constexpr long kMaxHeadDim = 256;
constexpr long kVecBytes = 256;
constexpr long kLocalResources = kSlices + 5 * kSlices * kSlots + 4;
constexpr long kSharedResources = 4;
constexpr long kMatQK = 1;
constexpr long kAtomicBytes = 128;
constexpr long kUramBytes = 1536 * 1024;
constexpr long kQBytes = kBr * kMaxHeadDim * sizeof(uint16_t);
constexpr long kOBytes = kBr * kMaxHeadDim * sizeof(float);
constexpr long kPVBytes = kOBytes;
constexpr long kScoreBytes = kBr * kBc * sizeof(float);
constexpr long kStateVectorBytes = kBr * 32;
constexpr long kMaskBytes = (kBr / 8) * (kBc / 8) * 32;
constexpr long kCtxQ = 0;
constexpr long kCtxO = kQBytes;
constexpr long kCtxTiles = kCtxO + kOBytes;
constexpr long kTileScore = 0;
constexpr long kTilePV = kTileScore + kScoreBytes;
constexpr long kTileM = kTilePV + kPVBytes;
constexpr long kTileL = kTileM + kStateVectorBytes;
constexpr long kTileAlpha = kTileL + kStateVectorBytes;
constexpr long kTileMask = kTileAlpha + kStateVectorBytes;
constexpr long kTileBytes = kTileMask + kMaskBytes;
constexpr long kContextBytes = kCtxTiles + kSlots * kTileBytes;
constexpr long kKBytes = kBc * kMaxHeadDim * sizeof(uint16_t);
constexpr long kVtBytes = kKBytes;
constexpr long kUramK0 = kSlices * kContextBytes;
constexpr long kUramVt0 = kUramK0 + 2 * kKBytes;
constexpr long kUramZero = kUramVt0 + 2 * kVtBytes;
static_assert(kUramZero + kVecBytes <= kUramBytes, "Attention URAM exceeds allocation");
static_assert(kLocalResources <= 32, "Attention exceeds the Local Resource window");

static inline uint64_t uram_addr(long offset) { return (uint64_t) dayan_uram_base() + offset; }
static inline long min_long(long a, long b) { return a < b ? a : b; }
static inline long extent(long n, long size) { return n <= 0 ? 0 : min_long(n, size); }
static inline void check(int status) {
    DAYAN_PARAMS_IF(status < 0) {
        DAYAN_PARAMS_ERROR();
    }
}

static inline dayan_resource_t q_resource(long slice) { return DAYAN_LOCAL_RESOURCE(slice); }
static inline dayan_resource_t score_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + slice * kSlots + slot); }
static inline dayan_resource_t p_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t pv_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 2 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t state_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 3 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t o_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 4 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t kv_resource(bool v, long slot) { return DAYAN_LOCAL_RESOURCE(kLocalResources - 4 + 2 * v + slot); }
static inline long tile_offset(long slice, long slot) {
    return slice * kContextBytes + kCtxTiles + slot * kTileBytes;
}

template <long Group> static inline void configure_lsu_rows(long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group> static inline void set_lsu_rows(long rows) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
}

template <long Group> static inline void configure_v_transpose(long head_dim, uint64_t row_stride) {
    constexpr long side = 32;

    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(head_dim / side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(row_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(kBc * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(kBc * side * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void load_v_transposed(uint64_t v_dram, long vt_base, long rows, uint64_t row_stride) {
    constexpr long side   = 32;
    const long     panels = (rows + side - 1) / side;

    for (long panel = 0; panel < panels; ++panel) {
        const long panel_rows = min_long(side, rows - panel * side);
        __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(panel_rows - 1);
        __tx_lsu_dma_d2u<Group>(uram_addr(vt_base) + panel * side * sizeof(uint16_t),
                                v_dram + static_cast<uint64_t>(panel * side) * row_stride);
    }
}

template <typename Format, long Group>
static inline void configure_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_M>(m - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(n - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_K>(k - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(Format::kMatDataType);
}


static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * result) {
    constexpr uint64_t max_u64 = ~uint64_t{ 0 };
    if (a != 0 && b > max_u64 / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t * result) {
    if (b > (~uint64_t{ 0 }) - a) {
        return false;
    }
    *result = a + b;
    return true;
}

static inline bool checked_strided_tensor_span(uint32_t   batches,
                                               uint32_t   heads,
                                               uint32_t   rows,
                                               uint64_t   row_bytes,
                                               uint64_t   batch_stride,
                                               uint64_t   head_stride,
                                               uint64_t   row_stride,
                                               uint64_t * result) {
    uint64_t batch_offset;
    uint64_t head_offset;
    uint64_t row_offset;
    uint64_t span;
    return checked_mul_u64(batches - 1U, batch_stride, &batch_offset) &&
           checked_mul_u64(heads - 1U, head_stride, &head_offset) &&
           checked_mul_u64(rows - 1U, row_stride, &row_offset) && checked_add_u64(batch_offset, head_offset, &span) &&
           checked_add_u64(span, row_offset, &span) && checked_add_u64(span, row_bytes, result);
}

static inline bool checked_tensor_bytes(uint32_t   d0,
                                        uint32_t   d1,
                                        uint32_t   d2,
                                        uint32_t   d3,
                                        uint64_t   element_bytes,
                                        uint64_t * result) {
    uint64_t value;
    return checked_mul_u64(d0, d1, &value) && checked_mul_u64(value, d2, &value) &&
           checked_mul_u64(value, d3, &value) && checked_mul_u64(value, element_bytes, result);
}

static inline bool valid_address_range(uint64_t base, uint64_t bytes) {
    return bytes <= (~uint64_t{ 0 }) - base;
}

static inline bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}



struct Cluster {
    dayan_cluster_barrier_t barrier;
    uint64_t uram[4];
    uint32_t owner[4];
};
static_assert(sizeof(Cluster) <= kAtomicBytes, "Attention shared metadata exceeds allocation");

struct State {
    const DayanFlashAttentionConfig * cfg;
    uint64_t q, k, v, out, uram;
    uint64_t blocks;
    uint32_t task, tasks, cores, lane, scale;
    long groups;
    bool split_kv;

    uint64_t work(uint64_t block) const { return task + block / groups * tasks; }
    uint32_t batch(uint64_t block) const { return (uint32_t) (work(block) / (split_kv ? cfg->kv_heads : cfg->q_heads)); }
    uint32_t head(uint64_t block) const {
        return split_kv ? (work(block) % cfg->kv_heads) * (cfg->q_heads / cfg->kv_heads) + block % groups * kBr : work(block) % cfg->q_heads;
    }
    long row(uint64_t block, long slice, long core) const {
        if (split_kv) {
            return 0;
        }
        return ((block % groups * kSlices + slice) * cores + core) * kBr;
    }
    long rows(uint64_t block, long slice) const {
        if (split_kv) {
            return slice ? 0 : extent(cfg->q_heads / cfg->kv_heads - block % groups * kBr, kBr);
        }
        return extent((long) cfg->q_len - row(block, slice, lane), kBr);
    }
    long key_begin(long core) const {
        const long visible = (cfg->flags & kFlagCausal) ? cfg->causal_offset + 1L : cfg->kv_len;
        return split_kv ? visible * core / 4 : 0;
    }
    long key_count(long core) const { return split_kv ? key_begin(core + 1) - key_begin(core) : cfg->kv_len; }
    long cols(long ki) const { return extent(key_count(lane) - ki * kBc, kBc); }
    long keys(uint64_t block, long slice, long core) const {
        if (split_kv) {
            return slice ? 0 : (key_count(core) + kBc - 1) / kBc;
        }
        const long first = row(block, slice, core);
        const long count = extent((long) cfg->q_len - first, kBr);
        if (!count) {
            return 0;
        }
        const long end = (cfg->flags & 1) ? cfg->causal_offset + first + count : cfg->kv_len;
        return (end + kBc - 1) / kBc;
    }
    long shared_keys(uint64_t block) const {
        if (split_kv) {
            return keys(block, 0, lane);
        }
        long count = 0;
        for (long slice = 0; slice < kSlices; ++slice) {
            for (long core = 0; core < cores; ++core) {
                const long n = keys(block, slice, core);
                count = n > count ? n : count;
            }
        }
        return count;
    }
    long users(uint64_t block, long ki) const {
        long count = 0;
        for (long slice = 0; slice < kSlices; ++slice) {
            count += ki < keys(block, slice, lane);
        }
        return count;
    }
    uint64_t q_address(uint64_t block, long slice) const {
        return q + batch(block) * cfg->q_batch_stride + head(block) * cfg->q_head_stride +
               row(block, slice, lane) * cfg->q_row_stride;
    }
    uint64_t kv_address(uint64_t block, long ki, bool value) const {
        const long kh = head(block) / (cfg->q_heads / cfg->kv_heads);
        const long first = key_begin(lane) + ki * kBc;
        return value ? v + batch(block) * cfg->v_batch_stride + kh * cfg->v_head_stride + first * cfg->v_row_stride :
                       k + batch(block) * cfg->k_batch_stride + kh * cfg->k_head_stride + first * cfg->k_row_stride;
    }
    uint64_t output_address(uint64_t block, long slice) const {
        return out + ((uint64_t) batch(block) * cfg->q_len * cfg->q_heads +
                      row(block, slice, lane) * cfg->q_heads + head(block)) * cfg->head_dim * sizeof(float);
    }
};

static inline long qk_swizzle(long head_dim) {
    return head_dim == 128 ? MAT_BATCH_SWIZZLE(5, 1, 5) :
           head_dim == 256 ? MAT_BATCH_SWIZZLE(5, 2, 5) : 0;
}
constexpr long kVSwizzle = MAT_BATCH_SWIZZLE(5, 1, 5);

static void load_q(void * argument) {
    const State & s = *(State *) argument;
    configure_lsu_rows<0>(s.cfg->head_dim * 2, s.split_kv ? s.cfg->q_head_stride : s.cfg->q_row_stride, s.cfg->head_dim * 2);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    for (uint64_t block = 0; block < s.blocks; ++block) {
        for (long slice = 0; slice < kSlices; ++slice) {
            const long rows = s.rows(block, slice);
            if (!rows) {
                continue;
            }
            check(dayan_resource_wait_free(q_resource(slice)));
            set_lsu_rows<0>(rows);
            __tx_lsu_dma_d2u<0>(uram_addr(slice * kContextBytes + kCtxQ), s.q_address(block, slice));
            check(dayan_resource_produce_lsu(q_resource(slice), 1, 0));
        }
    }
    dayan_wait_lsu();
}

template <int Cores, bool Value> static void load_kv(void * argument) {
    const State & s = *(State *) argument;
    uint64_t phases = 0;
    uint64_t generation = 0;
    if constexpr (Value) {
        configure_v_transpose<0>(s.cfg->head_dim, s.cfg->v_row_stride);
        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(kVSwizzle);
    } else {
        configure_lsu_rows<0>(s.cfg->head_dim * 2, s.cfg->k_row_stride, s.cfg->head_dim * 2);
        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    }
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        const uint64_t kv_base = s.kv_address(block, 0, Value);
        const uint64_t row_stride = Value ? s.cfg->v_row_stride : s.cfg->k_row_stride;
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const long slot = generation & 1;
            const long users = s.users(block, ki);
            const long cols = s.cols(ki);
            const long offset = Value ? kUramVt0 + slot * kVtBytes : kUramK0 + slot * kKBytes;
            const auto local = kv_resource(Value, slot);
            const auto ready = DAYAN_SHARED_RESOURCE(2 * Value + slot);
            const uint64_t shared_mask = DAYAN_RESOURCE_MASK(ready);
            int role = DAYAN_RESOURCE_ROLE_PRODUCER;
            if constexpr (Cores == 4) {
                role = dayan_resource_wait(users ? DAYAN_RESOURCE_MASK(local) | shared_mask : 0, shared_mask, phases, 0);
            } else if (users) {
                check(dayan_resource_wait_free(local));
            }
            check(role);
            if (!users) {
                if constexpr (Cores == 4) {
                    check(dayan_resource_consume_lsu(shared_mask));
                }
            } else if (role == DAYAN_RESOURCE_ROLE_PRODUCER) {
                if constexpr (Cores == 4) {
                    ((Cluster *) shared)->owner[2 * Value + slot] = s.lane;
                }
                if constexpr (Value) {
                    const long end = (cols + 31) / 32 * 64;
                    if (end < kBc * 2) {
                        configure_lsu_rows<1>(kBc * 2 - end, 0, kBc * 2);
                        set_lsu_rows<1>(s.cfg->head_dim);
                        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(kVSwizzle);
                        __tx_lsu_dma_u2u<1>(uram_addr(offset + end), uram_addr(kUramZero));
                    }
                    load_v_transposed<0>(kv_base + ki * kBc * row_stride, offset, cols, row_stride);
                } else {
                    set_lsu_rows<0>(cols);
                    __tx_lsu_dma_d2u<0>(uram_addr(offset), kv_base + ki * kBc * row_stride);
                }
                check(dayan_resource_produce_lsu(local, users, 0));
                if constexpr (Cores == 4) {
                    check(dayan_resource_produce_lsu(ready, Cores - 1, 0));
                }
            } else if (role == DAYAN_RESOURCE_ROLE_CONSUMER) {
                const Cluster & cluster = *(Cluster *) shared;
                const long owner = cluster.owner[2 * Value + slot];
                const long bytes = Value ? s.cfg->head_dim * kBc * 2 : cols * s.cfg->head_dim * 2;
                configure_lsu_rows<1>(bytes, bytes, bytes);
                __tx_lsu_dma_d2u<1>(uram_addr(offset), cluster.uram[owner] + offset);
                check(dayan_resource_produce_lsu(local, users, shared_mask));
            } else {
                DAYAN_PARAMS_ERROR();
            }
            phases ^= shared_mask;
        }
    }
    dayan_wait_lsu();
}

template <typename Format> static void calculate_qk(const State & s) {
    uint64_t phases = 0;
    uint64_t generation = 0;
    // Children allocate TCSR groups from zero; the root uses the remaining MAT group.
    configure_mat<Format, kMatQK>(kBr, kBc, s.cfg->head_dim, s.cfg->head_dim * 2, s.cfg->head_dim * 2, kBc * 4);
    __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const auto k = kv_resource(false, generation & 1);
            const long slot = ki & 1;
            const long users = s.users(block, ki);
            for (long slice = 0, used = 0; slice < kSlices; ++slice) {
                const long slice_keys = s.keys(block, slice, s.lane);
                if (ki >= slice_keys) {
                    continue;
                }
                const auto q = q_resource(slice);
                const uint64_t inputs = DAYAN_RESOURCE_MASK(q) | DAYAN_RESOURCE_MASK(k);
                // Both tokens protect the in-place score/P tile through the PV read.
                const uint64_t outputs = DAYAN_RESOURCE_MASK(score_resource(slice, slot)) | DAYAN_RESOURCE_MASK(p_resource(slice, slot));
                check(dayan_resource_wait(outputs, inputs, phases, 0));
                __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_M>(s.rows(block, slice) - 1);
                __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_N>(s.cols(ki) - 1);
                const long base = s.uram + slice * kContextBytes;
                __tx_mat_mm<kMatQK>(s.uram + tile_offset(slice, slot) + kTileScore, base + kCtxQ, s.uram + kUramK0 + (generation & 1) * kKBytes);
                const bool last = ki + 1 == slice_keys;
                // Q and K are reusable after their last MAT read, before softmax or PV.
                check(dayan_resource_produce_mat(score_resource(slice, slot), 1, DAYAN_RESOURCE_MASK(k) | (last ? DAYAN_RESOURCE_MASK(q) : 0)));
                if (last) {
                    phases ^= DAYAN_RESOURCE_MASK(q);
                }
                if (++used == users) {
                    phases ^= DAYAN_RESOURCE_MASK(k);
                }
            }
        }
    }
    dayan_wait_mat();
}

static inline void initialize_vec() {
    using namespace ggml_dayan_vec_detail;
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 8 * kBc * 4, 0, kBc * 4);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 16 * 32, 0);
}

template <bool Mask, bool First> static inline void online_softmax(long base, long previous, long rows, uint32_t scale) {
    using namespace ggml_dayan_vec_detail;
    // FlexVecAttentionSpec: 131 instructions, plus scale, constants, alpha store, and optional masks.
    constexpr long instructions = Mask ? 231 : 167;
    __tx_vec_tcsr_write<kVecDim2>((rows + 7) / 8 - 1);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(scale);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0xff7fffffU);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    for (long part = 0; part < 16; ++part) {
        __tx_vec_lsu_u2v<0, 0>(uram_addr(base + kTileScore + part * 32));
        __tx_vec_vv_fp32_mul<0, 0, 18, 0>();
        if constexpr (Mask) {
            __tx_vec_lsu_u2s<0, 3>(uram_addr(base + kTileMask + part * 32));
            __tx_vec_vv_mux2<0, 17, 0, 0>();
        }
        if (part) {
            __tx_vec_vv_fp32_max<0, Mask ? 3 : 2, 0, 0>();
        }
    }
    for (long amount = 4; amount; amount /= 2) {
        __tx_vec_shuffle_xor<1, 0>(amount);
        __tx_vec_vv_fp32_max<0, 0, 16, 0>();
    }
    if constexpr (First) {
        __tx_vec_lsu_g2v_broadcast32<1, 0>(0xff7fffffU);
    } else {
        __tx_vec_lsu_u2v<1, 1>(uram_addr(previous + kTileM));
    }
    __tx_vec_vv_fp32_max<1, 0, 16, 0>();
    // Q1 ages: new M=0, old M=1, zero=5, floor=6, scale=7.
    for (long part = 0; part < 16; ++part) {
        __tx_vec_lsu_u2v<0, 0>(uram_addr(base + kTileScore + part * 32));
        __tx_vec_vv_fp32_mul<0, 0, 23, 0>();
        __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
        __tx_vec_v_fp32_eru<0, 0, 0>();
        if constexpr (Mask) {
            __tx_vec_lsu_u2s<0, 3>(uram_addr(base + kTileMask + part * 32));
            __tx_vec_vv_mux2<0, 21, 0, 0>();
        }
        __tx_vec_lsu_v2u<0, 0>(uram_addr(base + kTileScore + part * 32));
        if (part) {
            __tx_vec_vv_fp32_add<0, Mask ? 5 : 4, 0, 0>();
        }
    }
    for (long amount = 4; amount; amount /= 2) {
        __tx_vec_shuffle_xor<1, 0>(amount);
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
    __tx_vec_vv_fp32_sub<0, 20, 19, 0>();
    __tx_vec_v_fp32_eru<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 2>(uram_addr(base + kTileAlpha));
    if constexpr (First) {
        __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    } else {
        __tx_vec_lsu_u2v<1, 1>(uram_addr(previous + kTileL));
    }
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_vv_fp32_add<0, 3, 0, 0>();
    __tx_vec_lsu_v2u<0, 1>(uram_addr(base + kTileL));
    __tx_vec_lsu_v2u<20, 1>(uram_addr(base + kTileM));
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static bool prepare_mask(const State & s, uint64_t block, long slice, long ki) {
    const long rows = s.rows(block, slice);
    const long cols = s.cols(ki);
    const bool causal = !s.split_kv && (s.cfg->flags & kFlagCausal);
    const long first = s.row(block, slice, s.lane) + s.cfg->causal_offset - ki * kBc;
    if (rows % 8 == 0 && cols == kBc && (!causal || first >= kBc - 1)) {
        return false;
    }
    uint64_t * masks = (uint64_t *) uram_addr(tile_offset(slice, ki & 1) + kTileMask);
    for (long group = 0; group < (rows + 7) / 8; ++group) {
        for (long part = 0; part < 16; ++part) {
            uint64_t mask = 0;
            for (long r = 0; r < 8; ++r) {
                const long row = group * 8 + r;
                const long valid = row >= rows ? 0 : causal ? extent(first + row + 1, cols) : cols;
                const long count = extent(valid - part * 8, 8);
                mask |= ((uint64_t{1} << count) - 1) << (r * 8);
            }
            masks[(group * 16 + part) * 4] = mask;
        }
    }
    return true;
}

static void softmax(void * argument) {
    const State & s = *(State *) argument;
    uint64_t phases = 0;
    uint64_t state_phases = 0;
    initialize_vec();
    for (uint64_t block = 0; block < s.blocks; ++block) {
        for (long ki = 0; ki < s.shared_keys(block); ++ki) {
            for (long slice = 0; slice < kSlices; ++slice) {
                if (ki >= s.keys(block, slice, s.lane)) {
                    continue;
                }
                const long slot = ki & 1;
                const uint64_t score = DAYAN_RESOURCE_MASK(score_resource(slice, slot));
                const uint64_t state = DAYAN_RESOURCE_MASK(state_resource(slice, slot));
                const uint64_t previous_state = ki == 0 ? 0 : DAYAN_RESOURCE_MASK(state_resource(slice, slot ^ 1));
                // Publication phases also advance for the final state, which has no softmax reader.
                check(dayan_resource_wait(state, score | previous_state, phases | (previous_state & ~state_phases), 0));
                const bool masked = prepare_mask(s, block, slice, ki);
                const long base = tile_offset(slice, slot);
                const long previous = tile_offset(slice, slot ^ 1);
                const long rows = s.rows(block, slice);
                if (ki == 0) {
                    if (masked) online_softmax<true, true>(base, previous, rows, s.scale);
                    else online_softmax<false, true>(base, previous, rows, s.scale);
                } else {
                    if (masked) online_softmax<true, false>(base, previous, rows, s.scale);
                    else online_softmax<false, false>(base, previous, rows, s.scale);
                }
                const bool last = ki + 1 == s.keys(block, slice, s.lane);
                check(dayan_resource_produce_vec(p_resource(slice, slot), 1, score));
                check(dayan_resource_produce_vec(state_resource(slice, slot), last ? 1 : 2, previous_state));
                phases ^= score;
                state_phases ^= state;
            }
        }
    }
    dayan_wait_vec();
}

template <typename Format> static void calculate_pv(void * argument) {
    const State & s = *(State *) argument;
    uint64_t phases = 0;
    uint64_t generation = 0;
    configure_mat<Format, 0>(kBr, s.cfg->head_dim, kBc, kBc * 4, kBc * 2, s.cfg->head_dim * 4);
    constexpr long value_type = Format::kMatDataType == BaseNPUInstBuilder::DT_BF16 ?
        BaseNPUInstBuilder::DT_MATIN_BF16 : BaseNPUInstBuilder::DT_MATIN_FP16;
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, value_type));
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(kVSwizzle);
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const auto v = kv_resource(true, generation & 1);
            const long slot = ki & 1;
            const long users = s.users(block, ki);
            for (long slice = 0, used = 0; slice < kSlices; ++slice) {
                if (ki >= s.keys(block, slice, s.lane)) {
                    continue;
                }
                const uint64_t p = DAYAN_RESOURCE_MASK(p_resource(slice, slot));
                const uint64_t inputs = p | DAYAN_RESOURCE_MASK(v);
                check(dayan_resource_wait(DAYAN_RESOURCE_MASK(pv_resource(slice, slot)), inputs, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(s.rows(block, slice) - 1);
                const long base = s.uram + tile_offset(slice, slot);
                __tx_mat_mm<0>(base + kTilePV, base + kTileScore, s.uram + kUramVt0 + (generation & 1) * kVtBytes);
                check(dayan_resource_produce_mat(pv_resource(slice, slot), 1, inputs));
                phases ^= p;
                if (++used == users) {
                    phases ^= DAYAN_RESOURCE_MASK(v);
                }
            }
        }
    }
    dayan_wait_mat();
}

template <bool First, bool Last> static inline void update_output(long base, long out) {
    using namespace ggml_dayan_vec_detail;
    constexpr long instructions = (First ? 2 : 6) + (Last ? 3 : 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_addr(base + kTilePV));
    if constexpr (!First) {
        __tx_vec_lsu_u2v<1, 0>(uram_addr(out));
        __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(base + kTileAlpha));
        __tx_vec_vv_fp32_mul<1, 17, 16, 0>();
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
    if constexpr (Last) {
        __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(base + kTileL));
        __tx_vec_v_fp32_eru<1, 16, 1>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 0>(uram_addr(out));
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static void merge_decode(const State & s, long base, long rows) {
    using namespace ggml_dayan_vec_detail;
    Cluster & cluster = *(Cluster *) shared;
    // The unused second Q slice holds peer state and merge weights.
    const long peer = tile_offset(1, 0);
    const long alpha = peer + kTileAlpha;
    const long beta = peer + kTileMask;
    const long state_bytes = (rows + 7) / 8 * kVecBytes;
    check(dayan_cluster_barrier(4, &cluster.barrier));
    if (s.lane == 0) {
        for (long core = 1; core < 4; ++core) {
            const long slot = (s.keys(0, 0, core) - 1) & 1;
            const uint64_t source = cluster.uram[core];
            configure_lsu_rows<0>(state_bytes, state_bytes, state_bytes);
            __tx_lsu_dma_d2u<0>(uram_addr(peer + kTileM), source + tile_offset(0, slot) + kTileM);
            __tx_lsu_dma_d2u<0>(uram_addr(peer + kTileL), source + tile_offset(0, slot) + kTileL);
            configure_lsu_rows<0>(rows * s.cfg->head_dim * 4, 0, 0);
            __tx_lsu_dma_d2u<0>(uram_addr(kContextBytes + kCtxO), source + kCtxO);
            dayan_wait_lsu();

            configure_vec_grid((rows + 7) / 8);
            configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
            begin_vec_block(16);
            __tx_vec_lsu_u2v<0, 0>(uram_addr(base + kTileM));
            __tx_vec_lsu_u2v<1, 0>(uram_addr(peer + kTileM));
            __tx_vec_vv_fp32_max<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(uram_addr(base + kTileM));
            __tx_vec_vv_fp32_sub<0, 1, 0, 0>();
            __tx_vec_v_fp32_eru<0, 0, 0>();
            __tx_vec_lsu_v2u<0, 0>(uram_addr(alpha));
            __tx_vec_vv_fp32_sub<1, 16, 2, 0>();
            __tx_vec_v_fp32_eru<1, 16, 0>();
            __tx_vec_lsu_v2u<16, 0>(uram_addr(beta));
            __tx_vec_lsu_u2v<0, 0>(uram_addr(base + kTileL));
            __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
            __tx_vec_lsu_u2v<1, 0>(uram_addr(peer + kTileL));
            __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(uram_addr(base + kTileL));
            start_vec_block(16);
            dayan_wait_vec();

            configure_vec_grid(rows, s.cfg->head_dim / 64);
            configure_vec_iterator<kVecV0Base>(0, 0, s.cfg->head_dim * 4, kVecBytes);
            configure_vec_iterator<kVecV2Base>(0, 0, 32, 0);
            begin_vec_block(8);
            __tx_vec_lsu_u2v<0, 0>(uram_addr(kCtxO));
            __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(alpha));
            __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
            __tx_vec_lsu_u2v<1, 0>(uram_addr(kContextBytes + kCtxO));
            __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(beta));
            __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(uram_addr(kCtxO));
            start_vec_block(8);
            dayan_wait_vec();
        }
        begin_vec_block(5);
        __tx_vec_lsu_u2v<0, 0>(uram_addr(kCtxO));
        __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(base + kTileL));
        __tx_vec_v_fp32_eru<1, 16, 1>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>(uram_addr(kCtxO));
        start_vec_block(5);
        dayan_wait_vec();
    }
    // Peer O and softmax state stay live until all remote reads finish.
    check(dayan_cluster_barrier(4, &cluster.barrier));
}

static void output(void * argument) {
    using namespace ggml_dayan_vec_detail;
    const State & s = *(State *) argument;
    uint64_t phases = 0;
    uint64_t o_phases = 0;
    for (uint64_t block = 0; block < s.blocks; ++block) {
        if (block == 0 || s.split_kv) {
            configure_vec_grid(1, s.cfg->head_dim / 64);
            configure_vec_iterator<kVecV0Base>(0, 0, s.cfg->head_dim * 4, kVecBytes);
            configure_vec_iterator<kVecV2Base>(0, 0, 32, 0);
            configure_lsu_rows<0>(s.cfg->head_dim * 4, s.cfg->head_dim * 4, s.cfg->q_heads * s.cfg->head_dim * 4);
        }
        for (long ki = 0; ki < s.shared_keys(block); ++ki) {
            for (long slice = 0; slice < kSlices; ++slice) {
                const long keys = s.keys(block, slice, s.lane);
                if (ki >= keys) {
                    continue;
                }
                const long slot = ki & 1;
                const auto o = o_resource(slice, slot);
                const uint64_t current_o = DAYAN_RESOURCE_MASK(o);
                const uint64_t previous_o = ki == 0 ? 0 : DAYAN_RESOURCE_MASK(o_resource(slice, slot ^ 1));
                const uint64_t inputs = DAYAN_RESOURCE_MASK(pv_resource(slice, slot)) | DAYAN_RESOURCE_MASK(state_resource(slice, slot));
                // Two version tokens serialize updates to the single O allocation.
                check(dayan_resource_wait(current_o, inputs | previous_o, phases | (previous_o & ~o_phases), 0));
                const long rows = s.rows(block, slice);
                const long base = tile_offset(slice, slot);
                const long out = slice * kContextBytes + kCtxO;
                const bool last = ki + 1 == keys;
                const bool normalize = last && !s.split_kv;
                __tx_vec_tcsr_write<kVecDim2>(rows - 1);
                if (ki == 0) {
                    if (normalize) update_output<true, true>(base, out);
                    else update_output<true, false>(base, out);
                } else {
                    if (normalize) update_output<false, true>(base, out);
                    else update_output<false, false>(base, out);
                }
                const uint64_t merge_state = last && s.split_kv ? DAYAN_RESOURCE_MASK(state_resource(slice, slot)) : 0;
                check(dayan_resource_produce_vec(o, 1, (inputs & ~merge_state) | previous_o));
                phases ^= inputs;
                o_phases ^= current_o;
                if (last) {
                    check(dayan_resource_wait_ready(o, (o_phases & current_o) ? 0 : 1));
                    if (s.split_kv) {
                        merge_decode(s, base, rows);
                        configure_lsu_rows<0>(s.cfg->head_dim * 4, s.cfg->head_dim * 4, s.cfg->head_dim * 4);
                    }
                    if (!s.split_kv || s.lane == 0) {
                        set_lsu_rows<0>(rows);
                        __tx_lsu_dma_u2d<0>(s.output_address(block, slice), uram_addr(out));
                    }
                    check(dayan_resource_consume_lsu(current_o | merge_state));
                    dayan_wait_lsu();
                }
            }
        }
    }
    dayan_wait_vec();
    dayan_wait_lsu();
}

enum FlashAttentionError {
    FA_INVALID_ARGUMENT = -1,
    FA_UNSUPPORTED_HEAD_DIM = -2,
    FA_UNSUPPORTED_HEAD_MAPPING = -3,
    FA_SIZE_OVERFLOW = -4,
    FA_UNALIGNED_V = -5,
    FA_OUTPUT_ALIAS = -6,
};

template <typename Format, int Cores, bool SplitKV = false>
static int run(const DayanFlashAttentionConfig * cfg, const uint16_t * q, const uint16_t * k, const uint16_t * v, float * out) {
    static_assert(Cores == 1 || Cores == 4, "Attention supports one or four cores");
    if (!cfg) {
        return FA_INVALID_ARGUMENT;
    }
    const uint32_t batch = cfg->batch;
    const uint32_t q_heads = cfg->q_heads;
    const uint32_t kv_heads = cfg->kv_heads;
    const uint32_t q_len = cfg->q_len;
    const uint32_t kv_len = cfg->kv_len;
    const uint32_t head_dim = cfg->head_dim;
    const uint32_t scale_bits = cfg->scale_bits;
    const uint32_t causal_offset = cfg->causal_offset;
    const uint32_t flags = cfg->flags;
    const uint64_t q_row_stride = cfg->q_row_stride;
    const uint64_t q_head_stride = cfg->q_head_stride;
    const uint64_t q_batch_stride = cfg->q_batch_stride;
    const uint64_t k_row_stride = cfg->k_row_stride;
    const uint64_t k_head_stride = cfg->k_head_stride;
    const uint64_t k_batch_stride = cfg->k_batch_stride;
    const uint64_t v_row_stride = cfg->v_row_stride;
    const uint64_t v_head_stride = cfg->v_head_stride;
    const uint64_t v_batch_stride = cfg->v_batch_stride;
    const uint32_t task            = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks           = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t scale_magnitude = scale_bits & 0x7fffffffU;
    if (!q || !k || !v || !out || batch == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        (flags & ~kFlagCausal) != 0 || ((flags & kFlagCausal) == 0 && causal_offset != 0) ||
        ((flags & kFlagCausal) != 0 && static_cast<uint64_t>(causal_offset) + q_len > kv_len) ||
        (scale_bits & 0x7f800000U) == 0x7f800000U || ((scale_bits & 0x80000000U) != 0 && scale_magnitude != 0) ||
        // The one-core instantiation also runs under the four-core task mode: the fused executor
        // keeps that mode for its barriers while attention runs data-parallel per task.
        (dayan_task_mode() != Cores && !(Cores == 1 && dayan_task_mode() == 4U)) ||
        (Cores == 4 && tasks != 4) || tasks == 0U || task >= tasks ||
        (Cores == 1 && dayan_task_dim(DAYAN_TASK_Y) != 1U) || !dayan_task_dim(DAYAN_TASK_Y) || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        return FA_INVALID_ARGUMENT;
    }
    if (head_dim == 0 || head_dim > kMaxHeadDim || (head_dim % kVecLanes) != 0) {
        return FA_UNSUPPORTED_HEAD_DIM;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(head_dim) * sizeof(uint16_t);
    if (q_row_stride < row_bytes || k_row_stride < row_bytes || v_row_stride < row_bytes ||
        ((q_row_stride | q_head_stride | q_batch_stride | k_row_stride | k_head_stride | k_batch_stride | v_row_stride |
          v_head_stride | v_batch_stride) &
         63U) != 0U) {
        return FA_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        return FA_UNSUPPORTED_HEAD_MAPPING;
    }
    if constexpr (SplitKV) {
        static_assert(Cores == 4, "Split-KV uses four cores");
        if (q_len != 1 || ((flags & kFlagCausal) ? causal_offset + 1L : kv_len) < 4) {
            return FA_INVALID_ARGUMENT;
        }
    }

    const float scale_log2e = u32_as_float(scale_bits) * 1.4426950408889634f;
    if ((float_as_u32(scale_log2e) & 0x7f800000U) == 0x7f800000U) {
        return FA_INVALID_ARGUMENT;
    }
    if ((reinterpret_cast<uint64_t>(v) & 63U) != 0) {
        return FA_UNALIGNED_V;
    }

    const uint64_t q_base_addr   = reinterpret_cast<uint64_t>(q);
    const uint64_t k_base_addr   = reinterpret_cast<uint64_t>(k);
    const uint64_t v_base_addr   = reinterpret_cast<uint64_t>(v);
    const uint64_t out_base_addr = reinterpret_cast<uint64_t>(out);
    uint64_t       q_span;
    uint64_t       k_span;
    uint64_t       v_span;
    uint64_t       out_bytes;
    uint64_t       traversal_generations;
    const uint64_t q_tiles  = (static_cast<uint64_t>(q_len) + kBr - 1) / kBr;
    const uint64_t q_groups = (q_tiles + (kSlices * Cores) - 1) / (kSlices * Cores);
    const uint64_t kv_tiles = (static_cast<uint64_t>(kv_len) + kBc - 1) / kBc;
    if (!checked_strided_tensor_span(batch, q_heads, q_len, row_bytes, q_batch_stride, q_head_stride, q_row_stride,
                                     &q_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, k_batch_stride, k_head_stride, k_row_stride,
                                     &k_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, v_batch_stride, v_head_stride, v_row_stride,
                                     &v_span) ||
        !checked_tensor_bytes(batch, q_heads, q_len, head_dim, sizeof(float), &out_bytes) ||
        !checked_mul_u64(batch, q_heads, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, q_groups, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, kv_tiles, &traversal_generations) ||
        !valid_address_range(q_base_addr, q_span) || !valid_address_range(k_base_addr, k_span) ||
        !valid_address_range(v_base_addr, v_span) || !valid_address_range(out_base_addr, out_bytes)) {
        return FA_SIZE_OVERFLOW;
    }
    if (ranges_overlap(out_base_addr, out_bytes, q_base_addr, q_span) ||
        ranges_overlap(out_base_addr, out_bytes, k_base_addr, k_span) ||
        ranges_overlap(out_base_addr, out_bytes, v_base_addr, v_span)) {
        return FA_OUTPUT_ALIAS;
    }
    if (!dayan_uram_base() || dayan_uram_size() < kUramBytes) {
        return FA_INVALID_ARGUMENT;
    }

    State s = {};
    s.cfg = cfg;
    s.split_kv = SplitKV;
    s.q = q_base_addr;
    s.k = k_base_addr;
    s.v = v_base_addr;
    s.out = out_base_addr;
    s.uram = (uint64_t) dayan_uram_base() & (2 * 1024 * 1024 - 1);
    if (s.uram + kUramBytes > 2 * 1024 * 1024) {
        return FA_INVALID_ARGUMENT;
    }
    s.cores = Cores;
    s.lane = Cores == 4 ? task : 0;
    s.task = Cores == 4 ? dayan_task_id(DAYAN_TASK_Y) : task;
    s.tasks = Cores == 4 ? dayan_task_dim(DAYAN_TASK_Y) : tasks;
    s.scale = float_as_u32(scale_log2e);
    s.groups = SplitKV ? (q_heads / kv_heads + kBr - 1) / kBr : q_groups;
    const uint64_t work = (uint64_t) batch * (SplitKV ? kv_heads : q_heads);
    s.blocks = (work > s.task ? (work - s.task + s.tasks - 1) / s.tasks : 0) * s.groups;
    for (long i = 0; i < kVecBytes / 4; ++i) {
        ((uint32_t *) uram_addr(kUramZero))[i] = 0;
    }
    if constexpr (Cores == 4) {
        if (!shared) {
            return FA_INVALID_ARGUMENT;
        }
        Cluster & cluster = *(Cluster *) shared;
        cluster.uram[s.lane] = (uint64_t) dayan_uram_base();
        check(dayan_cluster_barrier(4, &cluster.barrier));
    }
    dayan_thread_resources_t resources = {};
    resources.local_resource_count = kLocalResources;
    resources.shared_resource_count = Cores == 4 && !SplitKV ? kSharedResources : 0;
    resources.lsu_tcsr_count = 1;
    check(dayan_create_thread("attn-load-q", load_q, &s, &resources));
    resources.lsu_tcsr_count = 2;
    check(dayan_create_thread("attn-load-k", load_kv<SplitKV ? 1 : Cores, false>, &s, &resources));
    check(dayan_create_thread("attn-load-v", load_kv<SplitKV ? 1 : Cores, true>, &s, &resources));
    resources.lsu_tcsr_count = 0;
    resources.vec_tcsr_count = 1;
    check(dayan_create_thread("attn-softmax", softmax, &s, &resources));
    resources.vec_tcsr_count = 0;
    resources.mat_tcsr_count = 1;
    check(dayan_create_thread("attn-pv", calculate_pv<Format>, &s, &resources));
    resources.mat_tcsr_count = 0;
    resources.vec_tcsr_count = 1;
    resources.lsu_tcsr_count = 1;
    check(dayan_create_thread("attn-output", output, &s, &resources));
    calculate_qk<Format>(s);
    check(dayan_thread_join_all());
    if constexpr (Cores == 4) {
        check(dayan_cluster_barrier(4, &((Cluster *) shared)->barrier));
    }
    return 0;
}


} // namespace ggml_dayan_flash_attention_detail
