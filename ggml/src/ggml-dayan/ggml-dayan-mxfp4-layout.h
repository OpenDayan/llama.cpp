#pragma once

#include <stdint.h>

struct DayanMxfp4Tile {
    enum : int { m_alignment = 32, max_c_buffers = 3, uram_bytes = 1536 * 1024 };
    int m, n, k, input_stride, products, c_buffers;
    bool scales, post;

    uint64_t a_scale_stride() const { return (m * 4 + 63) & ~63ULL; }

    uint64_t accumulator_offset() const {
        return ((2ULL * m + products * n) * input_stride + 1023) & ~1023ULL;
    }

    uint64_t scratch_offset() const {
        const uint64_t scale_bytes = scales ? (2 * a_scale_stride() + products * n * 4ULL) * (k / 128) : 0;
        return (accumulator_offset() + c_buffers * (uint64_t) m * (n / 2) * 4 * products + scale_bytes + 63) & ~63ULL;
    }

    uint64_t bytes() const {
        return scratch_offset() + (post && products == 1 ? (uint64_t) m * n * 17 / 64 : 0) + 64;
    }

    static DayanMxfp4Tile make(uint64_t rows, uint64_t columns, uint64_t inner, bool scales, int products, bool post, uint64_t tasks = 4) {
        const uint64_t grid_rows = tasks == 4 && rows >= 64 ? 2 : 1;
        const uint64_t grid_cols = tasks / grid_rows;
        const uint64_t local_rows = (rows + grid_rows - 1) / grid_rows;
        DayanMxfp4Tile tile = {scales ? (products == 2 ? 256 : 512) : 128, 512, 128, scales ? 64 : 256, products, 3, scales, post};
        while (tile.m > m_alignment && local_rows < (uint64_t) tile.m) { tile.m /= 2; }
        if (tile.m == m_alignment) {
            uint64_t best = UINT64_MAX;
            for (int n = post ? 256 : 128; n <= 2048; n *= 2) {
                DayanMxfp4Tile candidate = tile;
                candidate.n = n;
                if (candidate.bytes() > uram_bytes) { continue; }
                const uint64_t work = (columns + grid_cols * n - 1) / (grid_cols * n) * n;
                if (work <= best) { best = work; tile.n = n; }
            }
        }
        if (tile.bytes() > uram_bytes) { tile.c_buffers = 2; }
        const int limit = scales ? 640 : 512;
        tile.k = inner >= (uint64_t) limit ? limit : (inner + 127) / 128 * 128;
        for (;;) {
            tile.input_stride = scales ? tile.k / 2 : tile.k * 2;
            if (tile.bytes() <= uram_bytes || tile.k == 128) { break; }
            tile.k -= 128;
        }
        return tile;
    }

};

struct DayanMxfp4Layout {
    uint64_t rows, columns, row_stride, scale_stride, scale_offset, bytes;

    static DayanMxfp4Layout make(uint64_t rows, uint64_t columns) {
        const uint64_t row_stride = (columns / 2 + 63) & ~63ULL;
        const uint64_t scale_stride = ((rows + 7) & ~7ULL) * 4;
        const uint64_t scale_offset = rows * row_stride;
        return {rows, columns, row_stride, scale_stride, scale_offset,
                scale_offset + ((columns + 127) / 128) * scale_stride + 64};
    }

    uint64_t scale_index(uint64_t row, uint64_t block) const {
        return block / 4 * scale_stride + row * 4 + block % 4;
    }
};

static inline void dayan_mxfp4_pack(const DayanMxfp4Layout & layout, const uint8_t * ggml, uint8_t * device) {
    for (uint64_t row = 0; row < layout.rows; ++row) {
        for (uint64_t block = 0; block < layout.columns / 32; ++block) {
            const uint8_t * input = ggml + (row * (layout.columns / 32) + block) * 17;
            uint8_t * output = device + row * layout.row_stride + block * 16;
            device[layout.scale_offset + layout.scale_index(row, block)] = input[0];
            for (int i = 0; i < 8; ++i) {
                output[i] = (input[1 + 2 * i] & 15) | ((input[2 + 2 * i] & 15) << 4);
                output[8 + i] = (input[1 + 2 * i] >> 4) | (input[2 + 2 * i] & 240);
            }
        }
    }
}

static inline void dayan_mxfp4_unpack(const DayanMxfp4Layout & layout, const uint8_t * device, uint8_t * ggml) {
    for (uint64_t row = 0; row < layout.rows; ++row) {
        for (uint64_t block = 0; block < layout.columns / 32; ++block) {
            const uint8_t * input = device + row * layout.row_stride + block * 16;
            uint8_t * output = ggml + (row * (layout.columns / 32) + block) * 17;
            output[0] = device[layout.scale_offset + layout.scale_index(row, block)];
            for (int i = 0; i < 8; ++i) {
                output[1 + 2 * i] = (input[i] & 15) | ((input[8 + i] & 15) << 4);
                output[2 + 2 * i] = (input[i] >> 4) | (input[8 + i] & 240);
            }
        }
    }
}
