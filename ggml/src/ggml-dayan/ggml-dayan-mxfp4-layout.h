#pragma once

#include <stdint.h>

struct DayanMxfp4Tile {
    enum : int { m_alignment = 32, uram_bytes = 1536 * 1024 };
    int m, n, k, input_stride, products;
    bool scales, post;

    uint64_t a_scale_stride() const { return (m * 4 + 63) & ~63ULL; }

    uint64_t accumulator_offset() const {
        return ((2ULL * m + products * n) * input_stride + 1023) & ~1023ULL;
    }

    uint64_t scratch_offset() const {
        const uint64_t scale_bytes = scales ? (2 * a_scale_stride() + products * n * 4ULL) * (k / 128) : 0;
        return (accumulator_offset() + (uint64_t) m * n * 4 * products + scale_bytes + 63) & ~63ULL;
    }

    uint64_t bytes() const {
        return scratch_offset() + (post ? (uint64_t) m * n * 17 / 64 : 0) + 64;
    }

    static DayanMxfp4Tile make(uint64_t rows, uint64_t columns, uint64_t inner, bool scales, int products, bool post) {
        DayanMxfp4Tile tile = {products == 2 ? 256 : 512, 512, scales && !post ? 640 : 512, 0, products, scales, post};
        tile.input_stride = scales ? tile.k / 2 : tile.k * 2;
        while (tile.bytes() > uram_bytes) { tile.m -= m_alignment; }
        if (rows < (uint64_t) tile.m) { tile.m = (rows + m_alignment - 1) & ~(uint64_t) (m_alignment - 1); }
        if (rows < 128) {
            tile.n = (columns + 3) / 4 > 3072 ? 3072 : (((columns + 3) / 4 + 255) & ~255ULL);
            while (tile.n > 256 && tile.bytes() > uram_bytes) { tile.n -= 256; }
            if (scales) {
                for (int k = 1024; k <= 4096 && inner > (uint64_t) tile.k; k *= 2) {
                    DayanMxfp4Tile wider = tile;
                    wider.k = k;
                    wider.input_stride = k / 2;
                    if (wider.bytes() > uram_bytes) { break; }
                    tile = wider;
                }
            }
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
