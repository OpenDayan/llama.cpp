#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace ggml_dayan_matmul_4c_detail {

struct Matmul4cShape {
    enum : int {
        kTileM       = 512,
        kTileN       = 384,
        kTileK       = 256,
        kSlices      = 4,
        kWorkers     = MATMUL_DAYAN_4C_WORKERS,
        kAtomicBytes = MATMUL_DAYAN_4C_ATOMIC_BYTES,
        kUramBytes   = MATMUL_DAYAN_4C_URAM_BYTES,
    };
};

struct Constants {
    enum : int {
        kDramLoad = 0,
        kRemoteCopy = 1,
        kCStore = 3,
        kTileM = Matmul4cShape::kTileM,
        kTileN = Matmul4cShape::kTileN,
        kTileK = Matmul4cShape::kTileK,
        kSlices = Matmul4cShape::kSlices,
        kClusterRows = 2,
        kClusterColumns = 2,
        kClusterCores = 4,
        kCoherentCacheLine = 64,
        kUramAlignment = 32,
        kUramWindowBase = 0x00800000,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

static_assert(Constants::kTileM == 512 && Constants::kTileN == 384,
              "4-Core matmul uses a 512x384 output tile per Core");
static_assert(Constants::kTileK == 256,
              "RTL matmul reaches peak throughput with tile K fixed at 256");
static_assert(Constants::kTileM % Constants::kSlices == 0 &&
                  Constants::kTileN % Constants::kSlices == 0,
              "both output dimensions must split evenly across MAT groups");
static_assert(Constants::kClusterRows == Constants::kClusterColumns,
              "shared sliced/buffered resources assume a square Core grid");
static_assert((Constants::kUramWindowStride &
               (Constants::kUramWindowStride - 1)) == 0,
              "URAM window stride must be a power of two");

static inline int min_int(int left, int right)
{
    return left < right ? left : right;
}

template <bool ProducerLoadsA> struct UramLayout {
    static constexpr int slice_offset(int slice)
    {
        return (ProducerLoadsA ? Constants::kTileN : Constants::kTileM) *
               slice / Constants::kSlices;
    }

    static constexpr int slice_size(int slice)
    {
        return slice_offset(slice + 1) - slice_offset(slice);
    }

    static constexpr int max_rows(int slice)
    {
        return ProducerLoadsA ? Constants::kTileM : slice_size(slice);
    }

    static constexpr int max_cols(int slice)
    {
        return ProducerLoadsA ? slice_size(slice) : Constants::kTileN;
    }

    static constexpr int a_buffer_count()
    {
        return ProducerLoadsA ? 2 : Constants::kSlices;
    }

    static constexpr int b_buffer_count()
    {
        return ProducerLoadsA ? Constants::kSlices : 2;
    }

    static constexpr uint64_t a_bytes()
    {
        return (uint64_t)(ProducerLoadsA ? Constants::kTileM
                                         : slice_size(0)) *
               Constants::kTileK * sizeof(uint16_t);
    }

    static constexpr uint64_t b_bytes()
    {
        return (uint64_t)(ProducerLoadsA ? slice_size(0)
                                         : Constants::kTileN) *
               Constants::kTileK * sizeof(uint16_t);
    }

    static constexpr uint64_t c_bytes()
    {
        return (uint64_t)max_rows(0) * max_cols(0) * sizeof(float);
    }

    static constexpr uint64_t a_base(int buffer)
    {
        return (uint64_t)buffer * a_bytes();
    }

    static constexpr uint64_t b_start()
    {
        return (uint64_t)a_buffer_count() * a_bytes();
    }

    static constexpr uint64_t b_base(int buffer)
    {
        return b_start() + (uint64_t)buffer * b_bytes();
    }

    static constexpr uint64_t c_start()
    {
        return b_start() + (uint64_t)b_buffer_count() * b_bytes();
    }

    static constexpr uint64_t c_base(int slice)
    {
        return c_start() + (uint64_t)slice * c_bytes();
    }

    static constexpr uint64_t total_bytes()
    {
        return c_start() + Constants::kSlices * c_bytes();
    }
};

using AProducerLayout = UramLayout<true>;
using BProducerLayout = UramLayout<false>;

static_assert(BProducerLayout::total_bytes() == 1441792U,
              "512x384x256 single-A/double-B layout must use 1.375 MiB");
static_assert(AProducerLayout::total_bytes() == 1507328U,
              "512x384x256 double-A/sliced-B layout size changed");
static_assert(AProducerLayout::total_bytes() <= Matmul4cShape::kUramBytes,
              "A-producer matmul URAM layout is too large");
static_assert(BProducerLayout::total_bytes() <= Matmul4cShape::kUramBytes,
              "B-producer matmul URAM layout is too large");

struct alignas(Constants::kCoherentCacheLine) SharedBuffer {
    dayan_shared_resource_t resource;
};

static_assert(sizeof(SharedBuffer) == Constants::kCoherentCacheLine,
              "each shared resource must occupy one coherent cache line");

struct ClusterShared {
    dayan_cluster_barrier_t barrier;
    int32_t init_status;
    int32_t selected_status;
    int32_t selected_expert;
    uint64_t uram_base[Constants::kClusterCores];
    SharedBuffer sliced[Constants::kClusterRows][Constants::kSlices];
    SharedBuffer buffered[Constants::kClusterRows][2];
};

static_assert(sizeof(ClusterShared) <= Matmul4cShape::kAtomicBytes,
              "cluster matmul shared state exceeds its atomic-cache request");

static inline ClusterShared &cluster_shared()
{
    return *static_cast<ClusterShared *>(shared);
}

template <bool ProducerLoadsA>
static inline SharedBuffer *shared_a(uint32_t row_lane, int index)
{
    return ProducerLoadsA ? &cluster_shared().buffered[row_lane][index]
                          : &cluster_shared().sliced[row_lane][index];
}

template <bool ProducerLoadsA>
static inline SharedBuffer *shared_b(uint32_t column_lane, int index)
{
    return ProducerLoadsA ? &cluster_shared().sliced[column_lane][index]
                          : &cluster_shared().buffered[column_lane][index];
}

struct LocalBuffer {
    dayan_resource_t resource;
    uint64_t uram;
};

struct TileConfig {
    int rows;
    int columns;
};

struct LsuConfig {
    int dim0;
    int dim1;
};

struct MatmulState {
    int m;
    int n;
    int k;
    int count_m;
    int count_n;
    int count_k;
    uint32_t lane;
    uint32_t row_lane;
    uint32_t column_lane;
    uint64_t a_base;
    uint64_t bt_base;
    uint64_t c_base;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
    uint64_t uram_base;
    LocalBuffer a[Constants::kSlices];
    LocalBuffer b[Constants::kSlices];
    dayan_resource_t load_issued[2];
    dayan_resource_t mat_turn[2];
    dayan_resource_t c[Constants::kSlices];
    uint64_t generation_count;
    TileConfig mat_config[Constants::kSlices];
    LsuConfig store_config;
};

struct MatJob {
    uint64_t generation;
    int tile_m;
    int tile_n;
    int tile_k;
    int slice;
};

struct MatWorkerArgs {
    MatmulState *state;
    uint32_t worker;
};

static inline uint32_t a_source_lane(uint32_t row_lane)
{
    return row_lane == 0U ? 0U : 3U;
}

static inline uint32_t b_source_lane(uint32_t column_lane)
{
    return column_lane == 0U ? 2U : 1U;
}

static inline int row_partition_begin(const MatmulState *state,
                                      uint32_t row_lane)
{
    return (int)((uint64_t)state->m * row_lane /
                 Constants::kClusterRows);
}

static inline int row_partition_end(const MatmulState *state,
                                    uint32_t row_lane)
{
    return (int)((uint64_t)state->m * (row_lane + 1U) /
                 Constants::kClusterRows);
}

static inline int rows_for_tile(const MatmulState *state, int tile_m,
                                uint32_t row_lane)
{
    const int row_base = row_partition_begin(state, row_lane) +
                         tile_m * Constants::kTileM;
    const int row_end = row_partition_end(state, row_lane);

    if (row_end <= row_base)
        return 0;
    return min_int(row_end - row_base, Constants::kTileM);
}

template <bool ProducerLoadsA>
static inline int rows_for_slice(const MatmulState *state, int tile_m,
                                 uint32_t row_lane, int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    if (ProducerLoadsA)
        return rows_for_tile(state, tile_m, row_lane);
    const int row_base = row_partition_begin(state, row_lane) +
                         tile_m * Constants::kTileM +
                         Layout::slice_offset(slice);
    const int row_end = row_partition_end(state, row_lane);

    if (row_end <= row_base)
        return 0;
    return min_int(row_end - row_base, Layout::slice_size(slice));
}

static inline int cols_for_tile(const MatmulState *state, int tile_n,
                                uint32_t column_lane)
{
    const int global_tile_n = tile_n * Constants::kClusterColumns +
                              (int)column_lane;
    const int column_base = global_tile_n * Constants::kTileN;

    if (state->n <= column_base)
        return 0;
    return min_int(state->n - column_base, Constants::kTileN);
}

template <bool ProducerLoadsA>
static inline int cols_for_slice(const MatmulState *state, int tile_n,
                                 uint32_t column_lane, int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    if (!ProducerLoadsA)
        return cols_for_tile(state, tile_n, column_lane);
    const int global_tile_n = tile_n * Constants::kClusterColumns +
                              (int)column_lane;
    const int column_base = global_tile_n * Constants::kTileN +
                            Layout::slice_offset(slice);

    if (state->n <= column_base)
        return 0;
    return min_int(state->n - column_base, Layout::slice_size(slice));
}

template <bool ProducerLoadsA>
static inline int active_slices(const MatmulState *state, int tile_m,
                                int tile_n, uint32_t row_lane,
                                uint32_t column_lane)
{
    int count = 0;

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        if (rows_for_slice<ProducerLoadsA>(state, tile_m, row_lane,
                                           slice) > 0 &&
            cols_for_slice<ProducerLoadsA>(state, tile_n, column_lane,
                                           slice) > 0)
            ++count;
    }
    return count;
}

static inline int k_for_tile(const MatmulState *state, int tile_k)
{
    const int k_base = tile_k * Constants::kTileK;

    if (state->k <= k_base)
        return 0;
    return min_int(state->k - k_base, Constants::kTileK);
}

static inline void decode_generation(const MatmulState *state,
                                     uint64_t generation, int *tile_m,
                                     int *tile_n, int *tile_k)
{
    const uint64_t outer = generation / (uint64_t)state->count_k;

    *tile_k = (int)(generation % (uint64_t)state->count_k);
    *tile_n = (int)(outer % (uint64_t)state->count_n);
    *tile_m = (int)(outer / (uint64_t)state->count_n);
}

static inline void require_status(int status)
{
    if (status < 0)
        dayan_proc_exit(status);
}

template <bool ProducerLoadsA>
static inline bool generation_has_local_work(const MatmulState *state,
                                             int tile_m, int tile_n)
{
    return active_slices<ProducerLoadsA>(
               state, tile_m, tile_n, state->row_lane,
               state->column_lane) > 0;
}

template <bool ProducerLoadsA, long LoadGroup>
static inline LsuConfig configure_loader(const MatmulState *state,
                                         bool loads_a, bool from_dram)
{
    using Layout = UramLayout<ProducerLoadsA>;

    long checksum = 0;
    constexpr int row_bytes =
        Constants::kTileK * (int)sizeof(uint16_t);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
    const int rows = loads_a ?
        (ProducerLoadsA ? Constants::kTileM : Layout::slice_size(0)) :
        (ProducerLoadsA ? Layout::slice_size(0) : Constants::kTileN);
    const long input_stride = from_dram ?
        (loads_a ? (long)state->a_stride : (long)state->bt_stride) :
        row_bytes;

    set_lsu_matrix<LoadGroup>(checksum, rows, row_bytes, input_stride,
                              row_bytes);
    __tx_lsu_tcsr_write<
        LoadGroup * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<
        LoadGroup * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(
            from_dram ? swizzle : 0);
    return {
        row_bytes / 64 - 1,
        rows - 1,
    };
}

template <long Tcsr>
static inline void configure_lsu_tcsr(int value, int *configured_value)
{
    if (value == *configured_value)
        return;
    __tx_lsu_tcsr_write<Tcsr>(value);
    *configured_value = value;
}

static inline void begin_resource(dayan_resource_t *resource,
                                  uint32_t consumers)
{
    require_status(dayan_resource_wait_free(resource));
    require_status(dayan_resource_alloc(resource, 1U, consumers));
}

static inline void begin_resource(dayan_shared_resource_t *resource,
                                  uint32_t consumers)
{
    require_status(dayan_shared_resource_wait_free(resource));
    require_status(dayan_shared_resource_alloc(resource, 1U, consumers));
}

template <long LoadGroup>
static inline void issue_dram_load(LocalBuffer *local,
                                   SharedBuffer *remote,
                                   uint32_t local_consumers,
                                   bool remote_active, uint64_t dram,
                                   uint64_t major_bytes,
                                   dayan_resource_t *load_issued,
                                   int *configured_dim0)
{
    if (local_consumers)
        begin_resource(&local->resource, local_consumers);
    if (remote_active)
        begin_resource(&remote->resource, 1U);

    uint64_t uram = local->uram;
    bool first_command = true;

    while (major_bytes) {
        uint64_t handled_bytes = major_bytes & ~0x3fULL;
        const uint64_t remaining_bytes = major_bytes & 0x3fULL;
        int dim0;
        uint64_t request_size;

        if (handled_bytes == 0) {
            handled_bytes = remaining_bytes;
            dim0 = 0;
            request_size = remaining_bytes;
        } else {
            dim0 = (int)(handled_bytes / 64U - 1U);
            request_size = 64U;
        }
        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM0>(
            dim0, configured_dim0);
        __tx_lsu_tcsr_write<
            LoadGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
                request_size - 1U);
        __tx_lsu_dma_d2u<LoadGroup>((long)uram, (long)dram);
        if (first_command && load_issued)
            require_status(dayan_resource_produce(load_issued));
        first_command = false;
        uram += handled_bytes;
        dram += handled_bytes;
        major_bytes -= handled_bytes;
    }
    dayan_wait_lsu();

    if (remote_active)
        require_status(dayan_shared_resource_produce(&remote->resource));
    if (local_consumers)
        require_status(dayan_resource_produce(&local->resource));
}

template <long LoadGroup>
static inline void issue_remote_copy(LocalBuffer *local,
                                     SharedBuffer *remote,
                                     uint32_t local_consumers, int rows,
                                     uint64_t source_uram,
                                     int *configured_dim1)
{
    begin_resource(&local->resource, local_consumers);
    require_status(dayan_shared_resource_wait_ready(&remote->resource));
    configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
        rows - 1, configured_dim1);
    __tx_lsu_dma_d2u<LoadGroup>((long)local->uram, (long)source_uram);
    dayan_wait_lsu();
    require_status(dayan_shared_resource_consume(&remote->resource));
    require_status(dayan_resource_produce(&local->resource));
}

template <bool ProducerLoadsA, long LoadGroup>
static void dram_load_a(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        LsuConfig *configured)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const bool signal_copy =
        generation_has_local_work<ProducerLoadsA>(state, tile_m, tile_n);
    const int real_k = k_for_tile(state, tile_k);
    bool issued = false;

    if (signal_copy)
        begin_resource(&state->load_issued[generation & 1U], 1U);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int local_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane);
        const int peer_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane ^ 1U);

        if (!real_rows || (!local_users && !peer_users))
            return;
        const int buffer_index = (int)(generation & 1U);
        LocalBuffer *local = &state->a[buffer_index];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, buffer_index);

        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
            real_rows - 1, &configured->dim1);
        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)tile_m * Constants::kTileM;
        const uint64_t a_dram =
            state->a_base + global_row * state->a_stride +
            (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

        issue_dram_load<LoadGroup>(
            local, shared_buffer, (uint32_t)local_users, peer_users > 0,
            a_dram, (uint64_t)real_k * sizeof(uint16_t),
            signal_copy ? &state->load_issued[generation & 1U] : nullptr,
            &configured->dim0);
        return;
    }

    const int columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int peer_columns = cols_for_tile(
        state, tile_n, state->column_lane ^ 1U);
    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        const int real_rows = rows_for_slice<ProducerLoadsA>(
            state, tile_m, state->row_lane, slice);
        const bool local_active = real_rows > 0 && columns > 0;
        const bool peer_active = real_rows > 0 && peer_columns > 0;
        LocalBuffer *local = &state->a[slice];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, slice);

        if (!local_active && !peer_active)
            continue;
        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
            real_rows - 1, &configured->dim1);

        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)tile_m * Constants::kTileM +
            (uint64_t)Layout::slice_offset(slice);
        const uint64_t a_dram =
            state->a_base + global_row * state->a_stride +
            (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);
        dayan_resource_t *load_issued =
            !issued && signal_copy ?
                &state->load_issued[generation & 1U] : nullptr;

        issue_dram_load<LoadGroup>(
            local, shared_buffer, local_active ? 1U : 0U, peer_active,
            a_dram, (uint64_t)real_k * sizeof(uint16_t), load_issued,
            &configured->dim0);
        if (load_issued)
            issued = true;
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void dram_load_b(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        LsuConfig *configured)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const bool signal_copy =
        generation_has_local_work<ProducerLoadsA>(state, tile_m, tile_n);
    const int real_k = k_for_tile(state, tile_k);
    bool issued = false;

    if (signal_copy)
        begin_resource(&state->load_issued[generation & 1U], 1U);
    if (ProducerLoadsA) {
        const int local_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int peer_rows = rows_for_tile(
            state, tile_m, state->row_lane ^ 1U);

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int real_columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);
            const bool local_active = real_columns > 0 && local_rows > 0;
            const bool peer_active = real_columns > 0 && peer_rows > 0;
            LocalBuffer *local = &state->b[slice];
            SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
                state->column_lane, slice);

            if (!local_active && !peer_active)
                continue;
            configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
                real_columns - 1, &configured->dim1);
            const uint64_t global_column =
                (uint64_t)(tile_n * Constants::kClusterColumns +
                           (int)state->column_lane) * Constants::kTileN +
                (uint64_t)Layout::slice_offset(slice);
            const uint64_t bt_dram =
                state->bt_base + global_column * state->bt_stride +
                (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);
            dayan_resource_t *load_issued =
                !issued && signal_copy ?
                    &state->load_issued[generation & 1U] : nullptr;

            issue_dram_load<LoadGroup>(
                local, shared_buffer, local_active ? 1U : 0U,
                peer_active, bt_dram,
                (uint64_t)real_k * sizeof(uint16_t), load_issued,
                &configured->dim0);
            if (load_issued)
                issued = true;
        }
        return;
    }

    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane, state->column_lane);
    const int peer_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane ^ 1U,
        state->column_lane);
    const int buffer_index = (int)(generation & 1U);
    LocalBuffer *local = &state->b[buffer_index];
    SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
        state->column_lane, buffer_index);

    if (!real_columns || (!local_users && !peer_users))
        return;
    configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
        real_columns - 1, &configured->dim1);

    const uint64_t global_column =
        (uint64_t)(tile_n * Constants::kClusterColumns +
                   (int)state->column_lane) * Constants::kTileN;
    const uint64_t bt_dram =
        state->bt_base + global_column * state->bt_stride +
        (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

    issue_dram_load<LoadGroup>(
        local, shared_buffer, (uint32_t)local_users, peer_users > 0,
        bt_dram, (uint64_t)real_k * sizeof(uint16_t),
        signal_copy ? &state->load_issued[generation & 1U] : nullptr,
        &configured->dim0);
}

template <bool ProducerLoadsA>
static void dram_loader(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const bool loads_a =
        state->lane == a_source_lane(state->row_lane);
    LsuConfig configured =
        configure_loader<ProducerLoadsA, Constants::kDramLoad>(
            state, loads_a, true);

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        if (loads_a) {
            dram_load_a<ProducerLoadsA, Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured);
        } else {
            dram_load_b<ProducerLoadsA, Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured);
        }
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void copy_a(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n,
                   int *configured_dim1)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const uint32_t source_lane = a_source_lane(state->row_lane);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int local_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane);

        if (!real_rows || !local_users)
            return;
        const int buffer_index = (int)(generation & 1U);
        const uint64_t source_uram =
            cluster_shared().uram_base[source_lane] +
            Layout::a_base(buffer_index);

        issue_remote_copy<LoadGroup>(
            &state->a[buffer_index],
            shared_a<ProducerLoadsA>(state->row_lane, buffer_index),
            (uint32_t)local_users, real_rows, source_uram,
            configured_dim1);
        return;
    }

    const int columns = cols_for_tile(state, tile_n, state->column_lane);

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        const int real_rows = rows_for_slice<ProducerLoadsA>(
            state, tile_m, state->row_lane, slice);

        if (!real_rows || !columns)
            continue;
        LocalBuffer *local = &state->a[slice];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, slice);

        const uint64_t source_uram =
            cluster_shared().uram_base[source_lane] +
            Layout::a_base(slice);

        issue_remote_copy<LoadGroup>(
            local, shared_buffer, 1U, real_rows, source_uram,
            configured_dim1);
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void copy_b(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n, int *configured_dim1)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const uint32_t source_lane = b_source_lane(state->column_lane);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int real_columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);

            if (!real_rows || !real_columns)
                continue;
            const uint64_t source_uram =
                cluster_shared().uram_base[source_lane] +
                Layout::b_base(slice);

            issue_remote_copy<LoadGroup>(
                &state->b[slice],
                shared_b<ProducerLoadsA>(state->column_lane, slice),
                1U, real_columns, source_uram, configured_dim1);
        }
        return;
    }

    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane, state->column_lane);
    const int buffer_index = (int)(generation & 1U);

    if (!real_columns || !local_users)
        return;
    LocalBuffer *local = &state->b[buffer_index];
    SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
        state->column_lane, buffer_index);

    const uint64_t source_uram =
        cluster_shared().uram_base[source_lane] +
        Layout::b_base(buffer_index);

    issue_remote_copy<LoadGroup>(
        local, shared_buffer, (uint32_t)local_users, real_columns,
        source_uram, configured_dim1);
}

template <bool ProducerLoadsA>
static void remote_copy(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const bool copies_a =
        state->lane != a_source_lane(state->row_lane);
    LsuConfig configured =
        configure_loader<ProducerLoadsA, Constants::kRemoteCopy>(
            state, copies_a, false);

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        (void)tile_k;
        if (!generation_has_local_work<ProducerLoadsA>(
                state, tile_m, tile_n))
            continue;
        dayan_resource_t *load_issued =
            &state->load_issued[generation & 1U];

        require_status(dayan_resource_wait_ready(load_issued));
        require_status(dayan_resource_consume(load_issued));
        if (copies_a) {
            copy_a<ProducerLoadsA, Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured.dim1);
        } else {
            copy_b<ProducerLoadsA, Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured.dim1);
        }
    }
}

template <typename Format, bool ProducerLoadsA, long GroupBase, int Slice>
static inline void configure_mat_group()
{
    using Layout = UramLayout<ProducerLoadsA>;

    constexpr int rows = Layout::max_rows(Slice);
    constexpr int columns = Layout::max_cols(Slice);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);

    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(rows - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(columns - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        Constants::kTileK - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSTRIDE>(
        columns * sizeof(float));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(
        Format::kMatDataType);
}

template <bool ProducerLoadsA, long StoreGroup>
static inline LsuConfig configure_c_store(const MatmulState *state)
{
    using Layout = UramLayout<ProducerLoadsA>;

    constexpr int rows = Layout::max_rows(0);
    constexpr int columns = Layout::max_cols(0);
    long checksum = 0;
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);

    set_lsu_matrix<StoreGroup>(
        checksum, rows, columns * (int)sizeof(float),
        columns * (int)sizeof(float), (long)state->c_stride);
    __tx_lsu_tcsr_write<
        StoreGroup * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(swizzle);
    return {
        columns * (int)sizeof(float) / 64 - 1,
        rows - 1,
    };
}

static inline void pass_mat_turn(MatmulState *state, uint32_t worker)
{
    dayan_resource_t *turn = &state->mat_turn[worker ^ 1U];

    require_status(dayan_resource_wait_free(turn));
    require_status(dayan_resource_alloc(turn, 0U, 1U));
}

template <typename Format, bool ProducerLoadsA, long GroupBase, int Slice>
static void execute_mat_job(MatmulState *state, const MatJob &job,
                            uint32_t worker)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const int real_rows = rows_for_slice<ProducerLoadsA>(
        state, job.tile_m, state->row_lane, Slice);
    const int real_columns = cols_for_slice<ProducerLoadsA>(
        state, job.tile_n, state->column_lane, Slice);
    const int a_buffer_index =
        ProducerLoadsA ? (int)(job.generation & 1U) : Slice;
    const int b_buffer_index =
        ProducerLoadsA ? Slice : (int)(job.generation & 1U);
    LocalBuffer *a = &state->a[a_buffer_index];
    LocalBuffer *b = &state->b[b_buffer_index];
    const uint64_t c_uram = state->uram_base + Layout::c_base(Slice);
    TileConfig *config = &state->mat_config[Slice];

    if (real_rows != config->rows) {
        __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(
            real_rows - 1);
        config->rows = real_rows;
    }
    if (real_columns != config->columns) {
        __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(
            real_columns - 1);
        config->columns = real_columns;
    }
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        k_for_tile(state, job.tile_k) - 1);
    if (job.tile_k == 0)
        __tx_mat_mm<GroupBase>((long)c_uram, (long)a->uram,
                               (long)b->uram);
    else
        __tx_mat_mma<GroupBase>((long)c_uram, (long)a->uram,
                                (long)b->uram);
    /*
     * The next worker may enqueue only after this command.  The MAT command
     * stream remains strictly ordered, while each worker waits for its own
     * completion token and at most two commands can be outstanding.
     */
    pass_mat_turn(state, worker);
    dayan_wait_mat();

    if (job.tile_k + 1 == state->count_k)
        require_status(dayan_resource_produce(&state->c[Slice]));

    require_status(dayan_resource_consume(&a->resource));
    require_status(dayan_resource_consume(&b->resource));
}

template <typename Format, bool ProducerLoadsA>
static void mat_worker(void *argument)
{
    MatWorkerArgs *args = (MatWorkerArgs *)argument;
    MatmulState *state = args->state;
    const uint32_t worker = args->worker;
    const uint64_t job_count =
        state->generation_count * Constants::kSlices;

    for (uint64_t sequence = worker; sequence < job_count;
         sequence += 2U) {
        MatJob job;
        job.generation = sequence / Constants::kSlices;
        job.slice = (int)(sequence % Constants::kSlices);
        decode_generation(state, job.generation, &job.tile_m,
                          &job.tile_n, &job.tile_k);
        const bool active =
            rows_for_slice<ProducerLoadsA>(
                state, job.tile_m, state->row_lane, job.slice) > 0 &&
            cols_for_slice<ProducerLoadsA>(
                state, job.tile_n, state->column_lane, job.slice) > 0;

        if (active) {
            const int a_buffer = ProducerLoadsA ?
                (int)(job.generation & 1U) : job.slice;
            const int b_buffer = ProducerLoadsA ?
                job.slice : (int)(job.generation & 1U);

            require_status(dayan_resource_wait_ready(
                &state->a[a_buffer].resource));
            require_status(dayan_resource_wait_ready(
                &state->b[b_buffer].resource));
            if (job.tile_k == 0) {
                dayan_resource_t *output = &state->c[job.slice];

                require_status(dayan_resource_wait_free(output));
                require_status(dayan_resource_alloc(output, 1U, 1U));
            }
        }

        dayan_resource_t *turn = &state->mat_turn[worker];

        require_status(dayan_resource_wait_ready(turn));
        require_status(dayan_resource_consume(turn));
        if (active) {
            switch (job.slice) {
            case 0:
                execute_mat_job<Format, ProducerLoadsA, 0, 0>(
                    state, job, worker);
                break;
            case 1:
                execute_mat_job<Format, ProducerLoadsA, 4, 1>(
                    state, job, worker);
                break;
            case 2:
                execute_mat_job<Format, ProducerLoadsA, 2, 2>(
                    state, job, worker);
                break;
            case 3:
                execute_mat_job<Format, ProducerLoadsA, 1, 3>(
                    state, job, worker);
                break;
            default:
                dayan_proc_exit(-22);
            }
        } else {
            pass_mat_turn(state, worker);
        }
    }
}

template <bool ProducerLoadsA>
static void issue_c_store(MatmulState *state, int tile_m, int tile_n,
                          int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const int real_rows = rows_for_slice<ProducerLoadsA>(
        state, tile_m, state->row_lane, slice);
    const int real_columns = cols_for_slice<ProducerLoadsA>(
        state, tile_n, state->column_lane, slice);
    require_status(dayan_resource_wait_ready(&state->c[slice]));
    configure_lsu_tcsr<Constants::kCStore * LSU_GROUPSZ + LSU_DIM1>(
        real_rows - 1, &state->store_config.dim1);
    configure_lsu_tcsr<Constants::kCStore * LSU_GROUPSZ + LSU_DIM0>(
        (int)((uint64_t)real_columns * sizeof(float) / 64U - 1U),
        &state->store_config.dim0);
    const uint64_t global_row =
        (uint64_t)row_partition_begin(state, state->row_lane) +
        (uint64_t)tile_m * Constants::kTileM +
        (ProducerLoadsA ? 0U : (uint64_t)Layout::slice_offset(slice));
    const uint64_t global_column =
        (uint64_t)(tile_n * Constants::kClusterColumns +
                   (int)state->column_lane) * Constants::kTileN +
        (ProducerLoadsA ? (uint64_t)Layout::slice_offset(slice) : 0U);
    const uint64_t c_uram = state->uram_base + Layout::c_base(slice);
    const uint64_t c_dram =
        state->c_base + global_row * state->c_stride +
        global_column * sizeof(float);

    __tx_lsu_dma_u2d<Constants::kCStore>((long)c_dram, (long)c_uram);
}

static inline void complete_c_store_batch(MatmulState *state,
                                          uint32_t active_mask)
{
    dayan_wait_lsu();
    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        if (active_mask & (1U << slice))
            require_status(dayan_resource_consume(&state->c[slice]));
    }
}

template <bool ProducerLoadsA>
static void c_store_worker(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const uint64_t output_count =
        (uint64_t)state->count_m * (uint64_t)state->count_n;

    for (uint64_t output = 0; output < output_count; ++output) {
        const int tile_m = (int)(output / (uint64_t)state->count_n);
        const int tile_n = (int)(output % (uint64_t)state->count_n);
        uint32_t active_mask = 0;
        int issued = 0;

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int rows = rows_for_slice<ProducerLoadsA>(
                state, tile_m, state->row_lane, slice);
            const int columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);

            if (!rows || !columns)
                continue;
            active_mask |= 1U << slice;
            issue_c_store<ProducerLoadsA>(state, tile_m, tile_n, slice);
            if (++issued == 2) {
                complete_c_store_batch(state, active_mask);
                active_mask = 0;
                issued = 0;
            }
        }
        if (active_mask)
            complete_c_store_batch(state, active_mask);
    }
}

static inline int config_valid(const DayanMatmulConfig *config)
{
    const uint64_t int_limit = 0x7ffff000ULL;

    if (!config || !config->m || !config->n || !config->k ||
        config->m > int_limit || config->n > int_limit ||
        config->k > int_limit || (config->n % 16U) != 0 ||
        config->a_stride < config->k * sizeof(uint16_t) ||
        config->bt_stride < config->k * sizeof(uint16_t) ||
        config->output_stride < config->n * sizeof(float) ||
        (config->output_stride & 63U) ||
        config->a_stride > 0x7fffffffffffffffULL ||
        config->bt_stride > 0x7fffffffffffffffULL ||
        config->output_stride > 0x7fffffffffffffffULL)
        return 0;
    return 1;
}

static inline int initialize_shared_buffers(SharedBuffer *buffers, int count)
{
    for (int index = 0; index < count; ++index) {
        const int status = dayan_shared_resource_init(
            &buffers[index].resource);

        if (status < 0)
            return status;
    }
    return 0;
}

static int initialize_cluster_shared(ClusterShared &cluster, uint32_t lane,
                                     uint64_t uram)
{
    int status = 0;

    if (lane == 0U) {
        cluster.init_status = 0;
        status = initialize_shared_buffers(
            &cluster.sliced[0][0],
            Constants::kClusterRows * Constants::kSlices);
        if (status == 0)
            status = initialize_shared_buffers(
                &cluster.buffered[0][0], Constants::kClusterRows * 2);
        cluster.init_status = status;
    }
    cluster.uram_base[lane] = uram;
    status = dayan_cluster_barrier(Constants::kClusterCores,
                                   &cluster.barrier);
    if (status < 0)
        return status;
    return cluster.init_status;
}

template <typename Format, bool ProducerLoadsA>
static int run_pipeline(const DayanMatmulConfig *config,
                        const uint16_t *a, const uint16_t *bt, float *c,
                        uint32_t lane, uint64_t physical_uram)
{
    using Layout = UramLayout<ProducerLoadsA>;

    int status = initialize_cluster_shared(
        cluster_shared(), lane, physical_uram);
    if (status < 0)
        return status;

    MatmulState state;
    state.m = (int)config->m;
    state.n = (int)config->n;
    state.k = (int)config->k;
    state.count_m =
        (state.m + Constants::kClusterRows * Constants::kTileM - 1) /
        (Constants::kClusterRows * Constants::kTileM);
    state.count_n =
        (state.n + Constants::kClusterColumns * Constants::kTileN - 1) /
        (Constants::kClusterColumns * Constants::kTileN);
    state.count_k = (state.k + Constants::kTileK - 1) /
                    Constants::kTileK;
    state.lane = lane;
    state.row_lane = lane >> 1;
    state.column_lane = lane & 1U;
    state.a_base = (uint64_t)(uintptr_t)a;
    state.bt_base = (uint64_t)(uintptr_t)bt;
    state.c_base = (uint64_t)(uintptr_t)c;
    state.a_stride = config->a_stride;
    state.bt_stride = config->bt_stride;
    state.c_stride = config->output_stride;
    state.uram_base = physical_uram &
                      (Constants::kUramWindowStride - 1U);
    state.generation_count =
        (uint64_t)state.count_m * (uint64_t)state.count_n *
        (uint64_t)state.count_k;

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        state.mat_config[slice].rows = Layout::max_rows(slice);
        state.mat_config[slice].columns = Layout::max_cols(slice);
        status = dayan_resource_init(&state.c[slice]);
        if (status < 0)
            return status;
    }
    for (int index = 0; index < Layout::a_buffer_count(); ++index) {
        status = dayan_resource_init(&state.a[index].resource);
        if (status < 0)
            return status;
        state.a[index].uram = state.uram_base + Layout::a_base(index);
    }
    for (int index = 0; index < Layout::b_buffer_count(); ++index) {
        status = dayan_resource_init(&state.b[index].resource);
        if (status < 0)
            return status;
        state.b[index].uram = state.uram_base + Layout::b_base(index);
    }
    for (int index = 0; index < 2; ++index) {
        status = dayan_resource_init(&state.load_issued[index]);
        if (status < 0)
            return status;
        status = dayan_resource_init(&state.mat_turn[index]);
        if (status < 0)
            return status;
    }
    status = dayan_resource_alloc(&state.mat_turn[0], 0U, 1U);
    if (status < 0)
        return status;

    configure_mat_group<Format, ProducerLoadsA, 0, 0>();
    configure_mat_group<Format, ProducerLoadsA, 4, 1>();
    configure_mat_group<Format, ProducerLoadsA, 2, 2>();
    configure_mat_group<Format, ProducerLoadsA, 1, 3>();
    state.store_config =
        configure_c_store<ProducerLoadsA, Constants::kCStore>(&state);
    MatWorkerArgs mat_args[2] = {
        {&state, 0U},
        {&state, 1U},
    };
    status = dayan_create_proc("dram-load", dram_loader<ProducerLoadsA>,
                               &state);
    if (status >= 0)
        status = dayan_create_proc("remote-copy",
                                   remote_copy<ProducerLoadsA>, &state);
    if (status >= 0)
        status = dayan_create_proc("mat-0", mat_worker<Format, ProducerLoadsA>,
                                   &mat_args[0]);
    if (status >= 0)
        status = dayan_create_proc("mat-1", mat_worker<Format, ProducerLoadsA>,
                                   &mat_args[1]);
    if (status >= 0)
        status = dayan_create_proc("c-store",
                                   c_store_worker<ProducerLoadsA>, &state);
    if (status < 0)
        return status;

    status = dayan_join_all();
    if (status < 0)
        return status;
    return dayan_cluster_barrier(Constants::kClusterCores,
                                 &cluster_shared().barrier);
}

template <typename Format>
static int run_matmul_pipeline_4c(const DayanMatmulConfig *config,
                                  const uint16_t *a, const uint16_t *bt,
                                  float *c)
{
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    if (!config_valid(config))
        return -101;
    if (!a || !bt || !c)
        return -102;
    if (dayan_task_mode() != 4U)
        return -103;
    if (dayan_task_dim(DAYAN_TASK_X) != 4U)
        return -104;
    if (dayan_task_dim(DAYAN_TASK_Y) != 1U)
        return -105;
    if (dayan_task_dim(DAYAN_TASK_Z) != 1U)
        return -106;
    if (lane >= 4U)
        return -107;
    if (!shared)
        return -108;

    void *uram = dayan_uram_base();
    const bool producer_loads_a = config->m < config->n;
    const uint64_t required_uram = producer_loads_a ?
        AProducerLayout::total_bytes() : BProducerLayout::total_bytes();
    if (!uram || dayan_uram_size() < required_uram)
        return -120;

    const uint64_t physical_uram = (uint64_t)(uintptr_t)uram;
    const uint64_t expected_window = Constants::kUramWindowBase +
        (uint64_t)lane * Constants::kUramWindowStride;
    if (physical_uram < expected_window ||
        physical_uram - expected_window >
            Constants::kUramWindowStride - required_uram)
        return -109;

    return producer_loads_a ?
        run_pipeline<Format, true>(config, a, bt, c, lane, physical_uram) :
        run_pipeline<Format, false>(config, a, bt, c, lane, physical_uram);
}

static inline int finish_epilogue_4c()
{
    return dayan_cluster_barrier(Constants::kClusterCores,
                                 &cluster_shared().barrier);
}

static inline bool moe_allocation_fits_4c(uint64_t element_size,
                                          uint64_t ne0, uint64_t ne1,
                                          uint64_t ne2)
{
    const uint64_t dimensions[3] = {ne0, ne1, ne2};
    uint64_t bytes = element_size;

    for (int index = 0; index < 3; ++index) {
        if (!dimensions[index] || dimensions[index] > UINT64_MAX / bytes)
            return false;
        bytes *= dimensions[index];
    }
    return true;
}

static inline bool mul_mat_id_config_valid_4c(
    const DayanMulMatIdConfig *config, uint64_t output_element_size)
{
    return config && config->k && config->n && config->experts &&
        config->experts <= INT32_MAX && config->experts_used &&
        config->tokens && config->activation_slots &&
        config->experts_used % config->activation_slots == 0U &&
        config->n % 16U == 0U &&
        moe_allocation_fits_4c(sizeof(uint16_t), config->k, config->n,
                               config->experts) &&
        moe_allocation_fits_4c(sizeof(uint16_t), config->k,
                               config->activation_slots, config->tokens) &&
        moe_allocation_fits_4c(sizeof(int32_t), config->experts_used,
                               config->tokens, 1U) &&
        moe_allocation_fits_4c(output_element_size, config->n,
                               config->experts_used, config->tokens);
}

static inline bool same_mul_mat_id_selection_4c(
    const DayanMulMatIdConfig &first, const DayanMulMatIdConfig &second)
{
    return first.k == second.k && first.experts == second.experts &&
        first.experts_used == second.experts_used &&
        first.tokens == second.tokens &&
        first.activation_slots == second.activation_slots;
}

static int read_selected_expert_4c(const DayanMulMatIdConfig *config,
                                   const int32_t *ids, uint64_t index,
                                   int32_t *expert)
{
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    ClusterShared &cluster = cluster_shared();

    if (lane == 0U) {
        cluster.selected_status = 0;
        const uint64_t scratch = (uint64_t)(uintptr_t)dayan_uram_base();
        if (!scratch || dayan_uram_size() < sizeof(int32_t)) {
            cluster.selected_status = -12;
        } else {
            long checksum = 0;
            set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
            ggml_dayan_matmul_detail::load_matrix<0>(
                sizeof(int32_t), 1, 64, 1, scratch,
                (uint64_t)(uintptr_t)(ids + index));
            dayan_wait_lsu();
            dayan_cache_invalidate((void *)(uintptr_t)scratch, 64);
            cluster.selected_expert = *(const int32_t *)(uintptr_t)scratch;
            if (cluster.selected_expert < 0 ||
                (uint32_t)cluster.selected_expert >= config->experts)
                cluster.selected_status = -22;
        }
    }
    const int status = dayan_cluster_barrier(Constants::kClusterCores,
                                             &cluster.barrier);
    if (status < 0)
        return status;
    *expert = cluster.selected_expert;
    return cluster.selected_status;
}

static inline DayanMatmulConfig moe_matmul_config_4c(
    const DayanMulMatIdConfig *config, uint64_t output_element_size)
{
    return {
        1U,
        config->n,
        config->k,
        (uint64_t)config->k * sizeof(uint16_t),
        (uint64_t)config->k * sizeof(uint16_t),
        (uint64_t)config->n * output_element_size,
    };
}

static inline uint64_t moe_activation_offset_4c(
    const DayanMulMatIdConfig *config, uint32_t token, uint32_t used)
{
    const uint32_t slot = used % config->activation_slots;
    return ((uint64_t)token * config->activation_slots + slot) * config->k;
}

static inline uint64_t moe_output_offset_4c(
    const DayanMulMatIdConfig *config, uint32_t token, uint32_t used)
{
    return ((uint64_t)token * config->experts_used + used) * config->n;
}

static int write_perf_4c(const DayanMatmulConfig *config,
                         DayanMatmulPerf *perf, uint64_t kernel_begin)
{
    if (dayan_task_id(DAYAN_TASK_X) != 0U)
        return 0;

    const uint64_t kernel_end = ggml_dayan_matmul_detail::read_cycle();
    const uint64_t physical_uram = (uint64_t)(uintptr_t)dayan_uram_base();
    DayanMatmulPerf *perf_uram = (DayanMatmulPerf *)(uintptr_t)physical_uram;
    perf_uram->magic = kDayanMatmulPerfMagic;
    perf_uram->m = config->m;
    perf_uram->n = config->n;
    perf_uram->k = config->k;
    perf_uram->kernel_cycles = kernel_end - kernel_begin;
    perf_uram->pipeline_cycles = kernel_end - kernel_begin;
    dayan_cache_flush(perf_uram, 64);

    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
    ggml_dayan_matmul_detail::store_matrix<0>(
        sizeof(*perf_uram), 1, 64, 1,
        physical_uram & (Constants::kUramWindowStride - 1U),
        (uint64_t)(uintptr_t)perf);
    dayan_wait_lsu();
    return 0;
}

template <typename Format>
static int run_matmul_4c(const DayanMatmulConfig *config,
                         const uint16_t *a, const uint16_t *bt, float *c,
                         DayanMatmulPerf *perf)
{
    if (!perf)
        return -22;
    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    const int status = run_matmul_pipeline_4c<Format>(config, a, bt, c);
    return status < 0 ? status : write_perf_4c(config, perf, kernel_begin);
}

struct PostprocessLayout {
    enum : uint64_t {
        kRows       = 128,
        kColumns    = 384,
        kF32Bytes   = kRows * kColumns * sizeof(float),
        kLowpBytes  = kRows * kColumns * sizeof(uint16_t),
        kInput0     = 0,
        kInput1     = kInput0 + kF32Bytes,
        kOutput     = kInput1 + kF32Bytes,
        kConstants  = kOutput + kLowpBytes,
        kTotalBytes = kConstants +
                      ggml_dayan_matmul_detail::Constants::kConstantCount *
                          ggml_dayan_matmul_detail::Constants::kConstantStride,
    };
};

static_assert(PostprocessLayout::kTotalBytes <= Matmul4cShape::kUramBytes,
              "4-Core matmul postprocess layout is too large");

static inline float *workspace_4c(DayanMatmulPerf *perf)
{
    return (float *)((char *)perf + kDayanMatmulWorkspaceOffset);
}

static inline uint64_t postprocess_uram(uint64_t offset)
{
    return (uint64_t)(uintptr_t)dayan_uram_base() + offset;
}

static inline void initialize_postprocess_constants()
{
    const uint64_t base = postprocess_uram(PostprocessLayout::kConstants);
    const uint64_t stride = ggml_dayan_matmul_detail::Constants::kConstantStride;
    *(volatile float *)(uintptr_t)(base + 0U * stride) = -1.4426950408889634f;
    *(volatile float *)(uintptr_t)(base + 1U * stride) = 1.0f;
    *(volatile float *)(uintptr_t)(base + 2U * stride) = 0.044715f;
    *(volatile float *)(uintptr_t)(base + 3U * stride) = 0.7978845608028654f;
    *(volatile float *)(uintptr_t)(base + 4U * stride) = -2.8853900817779268f;
    *(volatile float *)(uintptr_t)(base + 5U * stride) = 2.0f;
    *(volatile float *)(uintptr_t)(base + 6U * stride) = 0.5f;
    dayan_cache_flush((void *)(uintptr_t)base,
                      ggml_dayan_matmul_detail::Constants::kConstantCount * stride);
}

static inline ggml_dayan_vec_detail::ActivationConstants postprocess_constants()
{
    const uint64_t base = postprocess_uram(PostprocessLayout::kConstants);
    const uint64_t stride = ggml_dayan_matmul_detail::Constants::kConstantStride;
    return {
        base + 0U * stride,
        base + 1U * stride,
        base + 2U * stride,
        base + 3U * stride,
        base + 4U * stride,
        base + 5U * stride,
        base + 6U * stride,
    };
}

static inline int postprocess_rows(const DayanMatmulConfig *config)
{
    return (int)((config->m + PostprocessLayout::kRows - 1U) /
                 PostprocessLayout::kRows);
}

static inline int postprocess_columns(const DayanMatmulConfig *config)
{
    return (int)((config->n + PostprocessLayout::kColumns - 1U) /
                 PostprocessLayout::kColumns);
}

static inline void decode_postprocess_tile(const DayanMatmulConfig *config,
                                           uint64_t tile, int *row_base,
                                           int *column_base, int *rows,
                                           int *columns)
{
    const int column_tiles = postprocess_columns(config);
    const int row_tile = (int)(tile / (uint64_t)column_tiles);
    const int column_tile = (int)(tile % (uint64_t)column_tiles);
    *row_base = row_tile * (int)PostprocessLayout::kRows;
    *column_base = column_tile * (int)PostprocessLayout::kColumns;
    *rows = min_int((int)config->m - *row_base,
                    (int)PostprocessLayout::kRows);
    *columns = min_int((int)config->n - *column_base,
                       (int)PostprocessLayout::kColumns);
}

static inline void configure_postprocess_load(long group, uint64_t input_stride)
{
    long checksum = 0;
    if (group == 0) {
        set_lsu_matrix<0>(checksum, PostprocessLayout::kRows,
                          PostprocessLayout::kColumns * sizeof(float),
                          (long)input_stride,
                          PostprocessLayout::kColumns * sizeof(float));
    } else {
        set_lsu_matrix<1>(checksum, PostprocessLayout::kRows,
                          PostprocessLayout::kColumns * sizeof(float),
                          (long)input_stride,
                          PostprocessLayout::kColumns * sizeof(float));
    }
}

static inline void load_postprocess_tile(long group, uint64_t destination,
                                         const float *source, int rows,
                                         int columns)
{
    if (group == 0) {
        ggml_dayan_matmul_detail::load_matrix<0>(
            (uint64_t)columns * sizeof(float), rows,
            PostprocessLayout::kColumns * sizeof(float),
            PostprocessLayout::kRows, destination,
            (uint64_t)(uintptr_t)source);
    } else {
        ggml_dayan_matmul_detail::load_matrix<1>(
            (uint64_t)columns * sizeof(float), rows,
            PostprocessLayout::kColumns * sizeof(float),
            PostprocessLayout::kRows, destination,
            (uint64_t)(uintptr_t)source);
    }
}

static inline void configure_postprocess_store(uint64_t output_stride)
{
    long checksum = 0;
    set_lsu_matrix<3>(checksum, PostprocessLayout::kRows,
                      PostprocessLayout::kColumns * sizeof(uint16_t),
                      PostprocessLayout::kColumns * sizeof(uint16_t),
                      (long)output_stride);
}

static inline void store_postprocess_tile(uint16_t *destination, int rows,
                                          int columns)
{
    ggml_dayan_matmul_detail::store_matrix<3>(
        (uint64_t)columns * sizeof(uint16_t), rows,
        PostprocessLayout::kColumns * sizeof(uint16_t),
        PostprocessLayout::kRows,
        postprocess_uram(PostprocessLayout::kOutput),
        (uint64_t)(uintptr_t)destination);
    dayan_wait_lsu();
}

template <typename Format>
static int quantize_matrix_4c(const DayanMatmulConfig *config,
                              const float *input, uint16_t *output)
{
    if (!config || !input || !output || config->output_stride <
            config->n * sizeof(uint16_t))
        return -22;

    configure_postprocess_load(0, config->n * sizeof(float));
    configure_postprocess_store(config->output_stride);
    const uint64_t tile_count = (uint64_t)postprocess_rows(config) *
                                postprocess_columns(config);
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    for (uint64_t tile = lane; tile < tile_count;
         tile += Constants::kClusterCores) {
        int row_base;
        int column_base;
        int rows;
        int columns;
        decode_postprocess_tile(config, tile, &row_base, &column_base,
                                &rows, &columns);
        load_postprocess_tile(
            0, postprocess_uram(PostprocessLayout::kInput0),
            input + (uint64_t)row_base * config->n + column_base,
            rows, columns);
        dayan_wait_lsu();
        const ggml_dayan_vec_detail::F32Tensor input_tile = {
            postprocess_uram(PostprocessLayout::kInput0),
            (long)PostprocessLayout::kColumns * (long)sizeof(float)};
        ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
            input_tile, input_tile,
            {postprocess_uram(PostprocessLayout::kOutput),
             (long)PostprocessLayout::kColumns * (long)sizeof(uint16_t)},
            rows, columns);
        configure_postprocess_load(0, config->n * sizeof(float));
        store_postprocess_tile(
            (uint16_t *)((char *)output + (uint64_t)row_base *
                         config->output_stride) + column_base,
            rows, columns);
    }
    return finish_epilogue_4c();
}

template <typename Format, bool GeGLU>
static int glu_matrix_4c(const DayanMatmulConfig *config,
                         const float *gate, const float *up,
                         uint16_t *output)
{
    if (!config || !gate || !up || !output || config->output_stride <
            config->n * sizeof(uint16_t))
        return -22;

    initialize_postprocess_constants();
    configure_postprocess_load(0, config->n * sizeof(float));
    configure_postprocess_load(1, config->n * sizeof(float));
    configure_postprocess_store(config->output_stride);
    const uint64_t tile_count = (uint64_t)postprocess_rows(config) *
                                postprocess_columns(config);
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    for (uint64_t tile = lane; tile < tile_count;
         tile += Constants::kClusterCores) {
        int row_base;
        int column_base;
        int rows;
        int columns;
        decode_postprocess_tile(config, tile, &row_base, &column_base,
                                &rows, &columns);
        load_postprocess_tile(
            0, postprocess_uram(PostprocessLayout::kInput0),
            gate + (uint64_t)row_base * config->n + column_base,
            rows, columns);
        load_postprocess_tile(
            1, postprocess_uram(PostprocessLayout::kInput1),
            up + (uint64_t)row_base * config->n + column_base,
            rows, columns);
        dayan_wait_lsu();
        const ggml_dayan_vec_detail::F32Tensor gate_tile = {
            postprocess_uram(PostprocessLayout::kInput0),
            (long)PostprocessLayout::kColumns * (long)sizeof(float)};
        const ggml_dayan_vec_detail::F32Tensor up_tile = {
            postprocess_uram(PostprocessLayout::kInput1),
            (long)PostprocessLayout::kColumns * (long)sizeof(float)};
        if constexpr (GeGLU) {
            ggml_dayan_vec_detail::issue_geglu(
                gate_tile, up_tile, gate_tile, postprocess_constants(),
                rows, columns);
        } else {
            ggml_dayan_vec_detail::issue_swiglu(
                gate_tile, up_tile, gate_tile, postprocess_constants(),
                rows, columns);
        }
        dayan_wait_vec();
        ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
            gate_tile, gate_tile,
            {postprocess_uram(PostprocessLayout::kOutput),
             (long)PostprocessLayout::kColumns * (long)sizeof(uint16_t)},
            rows, columns);
        configure_postprocess_load(0, config->n * sizeof(float));
        store_postprocess_tile(
            (uint16_t *)((char *)output + (uint64_t)row_base *
                         config->output_stride) + column_base,
            rows, columns);
    }
    return finish_epilogue_4c();
}

template <typename Format>
static int run_matmul_quantized_4c(const DayanMatmulConfig *config,
                                   const uint16_t *a, const uint16_t *bt,
                                   uint16_t *output, DayanMatmulPerf *perf)
{
    if (!config || !perf)
        return -22;
    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    DayanMatmulConfig f32 = *config;
    f32.output_stride = config->n * sizeof(float);
    float *workspace = workspace_4c(perf);
    int status = run_matmul_pipeline_4c<Format>(&f32, a, bt, workspace);
    if (status == 0)
        status = quantize_matrix_4c<Format>(config, workspace, output);
    return status < 0 ? status : write_perf_4c(config, perf, kernel_begin);
}

template <typename Format, bool GeGLU>
static int run_matmul_glu_4c(const DayanMatmulConfig *config,
                             const uint16_t *a, const uint16_t *gate_bt,
                             const uint16_t *up_bt, uint16_t *output,
                             DayanMatmulPerf *perf)
{
    if (!config || !perf)
        return -22;
    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    DayanMatmulConfig f32 = *config;
    f32.output_stride = config->n * sizeof(float);
    float *gate = workspace_4c(perf);
    float *up = gate + config->m * config->n;
    int status = run_matmul_pipeline_4c<Format>(&f32, a, gate_bt, gate);
    if (status == 0)
        status = run_matmul_pipeline_4c<Format>(&f32, a, up_bt, up);
    if (status == 0)
        status = glu_matrix_4c<Format, GeGLU>(config, gate, up, output);
    return status < 0 ? status : write_perf_4c(config, perf, kernel_begin);
}

template <typename Format>
static int run_matmul_pair_4c(const DayanMatmulPairConfig *config,
                              const uint16_t *a, const uint16_t *first_bt,
                              const uint16_t *second_bt, float *first,
                              float *second, DayanMatmulPerf *perf)
{
    if (!config || !perf || config->first.m != config->second.m ||
        config->first.k != config->second.k)
        return -22;
    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    int status = run_matmul_pipeline_4c<Format>(
        &config->first, a, first_bt, first);
    if (status == 0)
        status = run_matmul_pipeline_4c<Format>(
            &config->second, a, second_bt, second);
    return status < 0 ? status :
        write_perf_4c(&config->second, perf, kernel_begin);
}

template <typename Format>
static int run_mul_mat_id_4c(const DayanMulMatIdConfig *config,
                             const uint16_t *activation,
                             const uint16_t *weight, const int32_t *ids,
                             float *output, DayanMatmulPerf *perf)
{
    if (!mul_mat_id_config_valid_4c(config, sizeof(float)) || !activation ||
        !weight || !ids || !output || !perf)
        return -22;

    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    const DayanMatmulConfig matmul = moe_matmul_config_4c(
        config, sizeof(float));
    const uint64_t weight_elements = (uint64_t)config->n * config->k;
    int status = 0;

    for (uint32_t token = 0; token < config->tokens && status == 0; ++token) {
        for (uint32_t used = 0; used < config->experts_used && status == 0;
             ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                config, ids, (uint64_t)token * config->experts_used + used,
                &expert);
            if (status == 0) {
                status = run_matmul_pipeline_4c<Format>(
                    &matmul,
                    activation + moe_activation_offset_4c(config, token, used),
                    weight + (uint64_t)expert * weight_elements,
                    output + moe_output_offset_4c(config, token, used));
            }
        }
    }
    return status < 0 ? status : write_perf_4c(&matmul, perf, kernel_begin);
}

template <typename Format>
static int run_mul_mat_id_pair_4c(const DayanMulMatIdPairConfig *config,
                                  const uint16_t *activation,
                                  const uint16_t *first_weight,
                                  const uint16_t *second_weight,
                                  const int32_t *ids, float *first_output,
                                  float *second_output,
                                  DayanMatmulPerf *perf)
{
    if (!config ||
        !mul_mat_id_config_valid_4c(&config->first, sizeof(float)) ||
        !mul_mat_id_config_valid_4c(&config->second, sizeof(float)) ||
        !same_mul_mat_id_selection_4c(config->first, config->second) ||
        !activation || !first_weight || !second_weight || !ids ||
        !first_output || !second_output || !perf)
        return -22;

    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    const DayanMatmulConfig first_matmul = moe_matmul_config_4c(
        &config->first, sizeof(float));
    const DayanMatmulConfig second_matmul = moe_matmul_config_4c(
        &config->second, sizeof(float));
    const uint64_t first_weight_elements =
        (uint64_t)config->first.n * config->first.k;
    const uint64_t second_weight_elements =
        (uint64_t)config->second.n * config->second.k;
    int status = 0;

    for (uint32_t token = 0; token < config->first.tokens && status == 0;
         ++token) {
        for (uint32_t used = 0;
             used < config->first.experts_used && status == 0; ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                &config->first, ids,
                (uint64_t)token * config->first.experts_used + used,
                &expert);
            const uint16_t *input = activation + moe_activation_offset_4c(
                &config->first, token, used);
            if (status == 0) {
                status = run_matmul_pipeline_4c<Format>(
                    &first_matmul, input,
                    first_weight + (uint64_t)expert * first_weight_elements,
                    first_output + moe_output_offset_4c(
                        &config->first, token, used));
            }
            if (status == 0) {
                status = run_matmul_pipeline_4c<Format>(
                    &second_matmul, input,
                    second_weight + (uint64_t)expert * second_weight_elements,
                    second_output + moe_output_offset_4c(
                        &config->second, token, used));
            }
        }
    }
    return status < 0 ? status :
        write_perf_4c(&second_matmul, perf, kernel_begin);
}

template <typename Format, bool GeGLU>
static int run_mul_mat_id_glu_4c(const DayanMulMatIdConfig *config,
                                 const uint16_t *activation,
                                 const uint16_t *gate_weight,
                                 const uint16_t *up_weight,
                                 const int32_t *ids, uint16_t *output,
                                 DayanMatmulPerf *perf)
{
    if (!mul_mat_id_config_valid_4c(config, sizeof(uint16_t)) ||
        !activation || !gate_weight || !up_weight || !ids || !output ||
        !perf)
        return -22;

    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    DayanMatmulConfig matmul = moe_matmul_config_4c(config, sizeof(float));
    DayanMatmulConfig epilogue = moe_matmul_config_4c(
        config, sizeof(uint16_t));
    const uint64_t weight_elements = (uint64_t)config->n * config->k;
    float *gate = workspace_4c(perf);
    float *up = gate + config->n;
    int status = 0;

    for (uint32_t token = 0; token < config->tokens && status == 0; ++token) {
        for (uint32_t used = 0; used < config->experts_used && status == 0;
             ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                config, ids, (uint64_t)token * config->experts_used + used,
                &expert);
            const uint16_t *input = activation + moe_activation_offset_4c(
                config, token, used);
            const uint64_t weight_offset =
                (uint64_t)expert * weight_elements;
            if (status == 0)
                status = run_matmul_pipeline_4c<Format>(
                    &matmul, input, gate_weight + weight_offset, gate);
            if (status == 0)
                status = run_matmul_pipeline_4c<Format>(
                    &matmul, input, up_weight + weight_offset, up);
            if (status == 0)
                status = glu_matrix_4c<Format, GeGLU>(
                    &epilogue, gate, up,
                    output + moe_output_offset_4c(config, token, used));
        }
    }
    return status < 0 ? status : write_perf_4c(&matmul, perf, kernel_begin);
}

template <typename Format, bool HasGate>
static int postprocess_attention_projection_4c(
    const DayanAttentionProjectionConfig *config, const float *norm,
    const int32_t *positions, const uint16_t *raw_output, uint16_t *output)
{
    using namespace ggml_dayan_matmul_detail;

    attention_reset_scratch();
    attention_initialize_constants(config);
    attention_load_bytes(attention_uram(kAttentionNorm), norm,
                         config->head_dim * sizeof(float));
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    const uint32_t position_axes = config->rope_mode == 8U ? 4U : 1U;
    const uint32_t projection_step = config->head_dim * (HasGate ? 2U : 1U);
    const uint64_t jobs = (uint64_t)config->tokens * config->heads;

    for (uint64_t job = lane; job < jobs; job += Constants::kClusterCores) {
        const uint32_t token = (uint32_t)(job / config->heads);
        const uint32_t head = (uint32_t)(job % config->heads);
        attention_load_i32(attention_uram(kAttentionPositions),
                           positions + token, position_axes,
                           (uint64_t)config->tokens * sizeof(int32_t));
        const uint16_t *source = raw_output +
            (uint64_t)token * config->matmul.output_stride /
                sizeof(uint16_t) +
            (uint64_t)head * projection_step;
        uint16_t *destination = output +
            ((uint64_t)head * config->tokens + token) * config->head_dim;
        postprocess_attention_head<Format>(config, source, destination);
    }
    return finish_epilogue_4c();
}

template <typename Format, bool HasGate>
static int run_attention_projection_4c(
    const DayanAttentionProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const float *norm, const int32_t *positions,
    uint16_t *raw_output, uint16_t *output, DayanMatmulPerf *perf)
{
    if (!ggml_dayan_matmul_detail::attention_projection_config_valid(
            config, HasGate) ||
        !activation || !weight || !norm || !positions || !raw_output ||
        !output || !perf)
        return -22;

    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    DayanMatmulConfig f32 = config->matmul;
    f32.output_stride = config->matmul.n * sizeof(float);
    float *workspace = workspace_4c(perf);
    int status = run_matmul_pipeline_4c<Format>(
        &f32, activation, weight, workspace);
    if (status == 0)
        status = quantize_matrix_4c<Format>(
            &config->matmul, workspace, raw_output);
    if (status == 0)
        status = postprocess_attention_projection_4c<Format, HasGate>(
            config, norm, positions, raw_output, output);
    return status < 0 ? status :
        write_perf_4c(&config->matmul, perf, kernel_begin);
}

template <typename Format>
static int run_value_projection_4c(
    const DayanAttentionProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, uint16_t *raw_output, uint16_t *output,
    DayanMatmulPerf *perf)
{
    using namespace ggml_dayan_matmul_detail;

    if (!attention_projection_config_valid(config, false) || !activation ||
        !weight || !raw_output || !output || !perf)
        return -22;

    const uint64_t kernel_begin = read_cycle();
    DayanMatmulConfig f32 = config->matmul;
    f32.output_stride = config->matmul.n * sizeof(float);
    float *workspace = workspace_4c(perf);
    int status = run_matmul_pipeline_4c<Format>(
        &f32, activation, weight, workspace);
    if (status == 0)
        status = quantize_matrix_4c<Format>(
            &config->matmul, workspace, raw_output);
    if (status == 0) {
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint64_t jobs = (uint64_t)config->tokens * config->heads;
        for (uint64_t job = lane; job < jobs;
             job += Constants::kClusterCores) {
            const uint32_t token = (uint32_t)(job / config->heads);
            const uint32_t head = (uint32_t)(job % config->heads);
            const uint16_t *source = raw_output +
                (uint64_t)token * config->matmul.output_stride /
                    sizeof(uint16_t) +
                (uint64_t)head * config->head_dim;
            uint16_t *destination = output +
                ((uint64_t)head * config->tokens + token) * config->head_dim;
            attention_copy_bytes(destination, source,
                                 config->head_dim * sizeof(uint16_t));
        }
        status = finish_epilogue_4c();
    }
    return status < 0 ? status :
        write_perf_4c(&config->matmul, perf, kernel_begin);
}

template <typename Format>
static int run_k_cache_projection_4c(
    const DayanKVProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const float *norm,
    const int32_t *positions_indices, uint16_t *raw_output, uint16_t *cache,
    DayanMatmulPerf *perf)
{
    using namespace ggml_dayan_matmul_detail;

    if (!kv_projection_config_valid(config) || !activation || !weight ||
        !norm || !positions_indices || !raw_output || !cache || !perf)
        return -22;

    const DayanAttentionProjectionConfig *projection = &config->projection;
    const uint64_t kernel_begin = read_cycle();
    DayanMatmulConfig f32 = projection->matmul;
    f32.output_stride = projection->matmul.n * sizeof(float);
    float *workspace = workspace_4c(perf);
    int status = run_matmul_pipeline_4c<Format>(
        &f32, activation, weight, workspace);
    if (status == 0)
        status = quantize_matrix_4c<Format>(
            &projection->matmul, workspace, raw_output);
    if (status == 0) {
        attention_reset_scratch();
        attention_initialize_constants(projection);
        attention_load_bytes(attention_uram(kAttentionNorm), norm,
                             projection->head_dim * sizeof(float));
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint32_t position_axes = projection->rope_mode == 8U ? 4U : 1U;
        const int32_t *indices = positions_indices +
            (uint64_t)position_axes * projection->tokens;
        const uint32_t cache_width = projection->heads * projection->head_dim;
        for (uint32_t token = 0; token < projection->tokens && status == 0;
             ++token) {
            attention_load_i32(attention_uram(kAttentionPositions),
                               indices + token, 1U, sizeof(int32_t));
            const int32_t cache_row =
                *attention_uram_pointer<int32_t>(kAttentionPositions);
            if (cache_row < 0 || (uint32_t)cache_row >= config->cache_rows) {
                status = -22;
                break;
            }
            attention_load_i32(attention_uram(kAttentionPositions),
                               positions_indices + token, position_axes,
                               (uint64_t)projection->tokens * sizeof(int32_t));
            for (uint32_t head = lane; head < projection->heads;
                 head += Constants::kClusterCores) {
                const uint16_t *source = raw_output +
                    (uint64_t)token * projection->matmul.output_stride /
                        sizeof(uint16_t) +
                    (uint64_t)head * projection->head_dim;
                uint16_t *destination = cache +
                    (uint64_t)cache_row * cache_width +
                    (uint64_t)head * projection->head_dim;
                postprocess_attention_head<Format>(
                    projection, source, destination);
            }
            status = finish_epilogue_4c();
        }
    }
    return status < 0 ? status :
        write_perf_4c(&projection->matmul, perf, kernel_begin);
}

template <typename Format>
static int run_v_cache_projection_4c(
    const DayanKVProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const int32_t *indices, uint16_t *raw_output,
    uint16_t *cache, DayanMatmulPerf *perf)
{
    using namespace ggml_dayan_matmul_detail;

    if (!kv_projection_config_valid(config) || !activation || !weight ||
        !indices || !raw_output || !cache || !perf)
        return -22;

    const DayanAttentionProjectionConfig *projection = &config->projection;
    const uint64_t kernel_begin = read_cycle();
    DayanMatmulConfig f32 = projection->matmul;
    f32.output_stride = projection->matmul.n * sizeof(float);
    float *workspace = workspace_4c(perf);
    int status = run_matmul_pipeline_4c<Format>(
        &f32, activation, weight, workspace);
    if (status == 0)
        status = quantize_matrix_4c<Format>(
            &projection->matmul, workspace, raw_output);
    if (status == 0) {
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint32_t cache_width = projection->heads * projection->head_dim;
        for (uint32_t token = 0; token < projection->tokens && status == 0;
             ++token) {
            attention_load_i32(attention_uram(kAttentionPositions),
                               indices + token, 1U, sizeof(int32_t));
            const int32_t cache_row =
                *attention_uram_pointer<int32_t>(kAttentionPositions);
            if (cache_row < 0 || (uint32_t)cache_row >= config->cache_rows) {
                status = -22;
                break;
            }
            for (uint32_t head = lane; head < projection->heads;
                 head += Constants::kClusterCores) {
                const uint16_t *source = raw_output +
                    (uint64_t)token * projection->matmul.output_stride /
                        sizeof(uint16_t) +
                    (uint64_t)head * projection->head_dim;
                uint16_t *destination = cache +
                    (uint64_t)cache_row * cache_width +
                    (uint64_t)head * projection->head_dim;
                attention_copy_bytes(destination, source,
                                     projection->head_dim * sizeof(uint16_t));
            }
            status = finish_epilogue_4c();
        }
    }
    return status < 0 ? status :
        write_perf_4c(&projection->matmul, perf, kernel_begin);
}

template <typename Format>
static inline void issue_gated_lowp_to_f32_4c(uint64_t input, int rows,
                                               int columns)
{
    constexpr long instructions = 3;
    const long row_stride = PostprocessLayout::kColumns * sizeof(float);

    ggml_dayan_vec_detail::configure_vec_grid(
        columns / ggml_dayan_matmul_detail::Constants::kVecLanes, rows);
    ggml_dayan_matmul_detail::configure_vec_iterator<
        ggml_dayan_matmul_detail::kVecV0Base>(
        0, 0, ggml_dayan_matmul_detail::Constants::kVecBytes, row_stride);
    ggml_dayan_matmul_detail::configure_vec_iterator<
        ggml_dayan_matmul_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_matmul_detail::configure_vec_iterator<
        ggml_dayan_matmul_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_matmul_detail::configure_vec_iterator<
        ggml_dayan_matmul_detail::kVecV3Base>(
        0, 0, ggml_dayan_matmul_detail::Constants::kVecBytes, row_stride);
    ggml_dayan_matmul_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16)
        ggml_dayan_vec_detail::vec_cvt_fp32_bf16<0, 0, 0>();
    else
        ggml_dayan_vec_detail::vec_cvt_fp32_fp16<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(input);
    ggml_dayan_matmul_detail::start_vec_block(instructions);
}

template <typename Format>
static int gate_attention_input_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const uint16_t *gate, uint16_t *output)
{
    using namespace ggml_dayan_matmul_detail;

    const uint32_t width = config->heads * config->head_dim;
    const uint32_t heads_per_tile =
        PostprocessLayout::kColumns / config->head_dim;
    const uint32_t row_tiles =
        (config->tokens + PostprocessLayout::kRows - 1U) /
        PostprocessLayout::kRows;
    const uint32_t head_tiles =
        (config->heads + heads_per_tile - 1U) / heads_per_tile;
    const uint64_t tile_count = (uint64_t)row_tiles * head_tiles;
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    initialize_postprocess_constants();
    configure_postprocess_load(0, (uint64_t)width * sizeof(float));
    configure_postprocess_store((uint64_t)width * sizeof(uint16_t));
    for (uint64_t tile = lane; tile < tile_count;
         tile += Constants::kClusterCores) {
        const uint32_t row_tile = (uint32_t)(tile / head_tiles);
        const uint32_t head_tile = (uint32_t)(tile % head_tiles);
        const uint32_t row_base = row_tile * PostprocessLayout::kRows;
        const uint32_t head_base = head_tile * heads_per_tile;
        const int rows = min_int(
            (int)(config->tokens - row_base),
            (int)PostprocessLayout::kRows);
        const uint32_t heads = (uint32_t)min_int(
            (int)(config->heads - head_base), (int)heads_per_tile);
        const int columns = heads * config->head_dim;

        load_postprocess_tile(
            0, postprocess_uram(PostprocessLayout::kInput0),
            attention + (uint64_t)row_base * width +
                (uint64_t)head_base * config->head_dim,
            rows, columns);
        configure_gated_lsu<1>(
            0, config->head_dim - 1U, rows - 1U, sizeof(uint16_t),
            sizeof(uint16_t), 2U * width * sizeof(uint16_t), sizeof(float),
            sizeof(float), PostprocessLayout::kColumns * sizeof(float), 6,
            sizeof(uint16_t) - 1U, sizeof(float) - 1U);
        __tx_lsu_tcsr_write<1 * LSU_GROUPSZ + LSU_DIM2>(heads - 1U);
        __tx_lsu_tcsr_write<1 * LSU_GROUPSZ + LSU_INDIM2STRIDE>(
            2U * config->head_dim * sizeof(uint16_t));
        __tx_lsu_tcsr_write<1 * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(
            config->head_dim * sizeof(float));
        __tx_lsu_dma_d2u<1>(
            postprocess_uram(PostprocessLayout::kInput1),
            reinterpret_cast<uint64_t>(
                gate + (uint64_t)row_base * 2U * width +
                (uint64_t)head_base * 2U * config->head_dim +
                config->head_dim));
        dayan_wait_lsu();
        issue_gated_lowp_to_f32_4c<Format>(
            postprocess_uram(PostprocessLayout::kInput1), rows, columns);
        dayan_wait_vec();
        ggml_dayan_vec_detail::issue_sigmoid_mul(
            {postprocess_uram(PostprocessLayout::kInput1),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            {postprocess_uram(PostprocessLayout::kInput0),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            {postprocess_uram(PostprocessLayout::kInput0),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            postprocess_constants(), rows, columns);
        dayan_wait_vec();
        const ggml_dayan_vec_detail::F32Tensor attention_tile = {
            postprocess_uram(PostprocessLayout::kInput0),
            (long)PostprocessLayout::kColumns * (long)sizeof(float)};
        ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
            attention_tile, attention_tile,
            {postprocess_uram(PostprocessLayout::kOutput),
             (long)PostprocessLayout::kColumns * (long)sizeof(uint16_t)},
            rows, columns);
        configure_postprocess_load(0, (uint64_t)width * sizeof(float));
        store_postprocess_tile(
            output + (uint64_t)row_base * width +
                (uint64_t)head_base * config->head_dim,
            rows, columns);
    }
    return finish_epilogue_4c();
}

template <typename Format>
static int run_gated_output_projection_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const uint16_t *gate, const uint16_t *weight, float *output,
    DayanMatmulPerf *perf)
{
    const uint64_t width = config ?
        (uint64_t)config->heads * config->head_dim : 0U;
    if (!config || !attention || !gate || !weight || !output || !perf ||
        !config->tokens || !config->heads || !config->head_dim ||
        config->matmul.m != config->tokens || config->matmul.k != width ||
        config->head_dim > PostprocessLayout::kColumns ||
        config->head_dim % ggml_dayan_matmul_detail::Constants::kVecLanes !=
            0U ||
        config->matmul.m >= config->matmul.n)
        return -22;

    const uint64_t kernel_begin = ggml_dayan_matmul_detail::read_cycle();
    uint16_t *gated = reinterpret_cast<uint16_t *>(workspace_4c(perf));
    int status = gate_attention_input_4c<Format>(
        config, attention, gate, gated);
    if (status == 0)
        status = run_matmul_pipeline_4c<Format>(
            &config->matmul, gated, weight, output);
    return status < 0 ? status :
        write_perf_4c(&config->matmul, perf, kernel_begin);
}

}  // namespace ggml_dayan_matmul_4c_detail

#endif
