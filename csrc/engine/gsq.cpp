// The IQ formats' decode (xtier/engine/gsq.h): llama.cpp's dequantize_row_iq* semantics, in plain C++.
//
// Ported from ggml/src/ggml-quants.c (MIT, (c) the llama.cpp authors): the grid tables are in gsq_tables.h, the
// rest is the reference code's arithmetic.
#include "gsq.h"

#include <cstring>

#include "gsq_tables.h"

namespace xtier {
namespace {

constexpr float IQ1S_DELTA = 0.125f;   // llama.cpp's IQ1S_DELTA

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {   // subnormal: normalize it
            uint32_t e = 0, m = man;
            while (!(m & 0x400)) { m <<= 1; e++; }
            bits = sign | ((113 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);   // inf or NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline uint32_t u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint16_t u16(const uint8_t *p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace

int gsq_row_bytes(int type, int n) {
    int block;
    switch (type) {
        case GSQ_IQ2_XXS: block = 66; break;
        case GSQ_IQ2_XS: block = 74; break;
        case GSQ_IQ1_S: block = 50; break;
        case GSQ_IQ1_M: block = 56; break;
        default: return 0;
    }
    if (n <= 0 || n % GSQ_BLOCK_VALUES != 0) return 0;
    return (n / GSQ_BLOCK_VALUES) * block;
}

void gsq_dequant_row(int type, const uint8_t *src, float *dst, int n) {
    const int nb = n / GSQ_BLOCK_VALUES;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = src + size_t(i) * gsq_row_bytes(type, GSQ_BLOCK_VALUES);
        float *y = dst + size_t(i) * GSQ_BLOCK_VALUES;
        if (type == GSQ_IQ2_XXS) {
            const float d = fp16_to_fp32(u16(b));
            for (int ib32 = 0; ib32 < GSQ_BLOCK_VALUES / 32; ++ib32) {
                uint32_t aux32[2];
                std::memcpy(aux32, b + 2 + 8 * ib32, 2 * sizeof(uint32_t));
                const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
                const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux32);
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2xxs_grid + aux8[l]);
                    const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                    for (int j = 0; j < 8; ++j) {
                        y[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                y += 32;
            }
        } else if (type == GSQ_IQ2_XS) {
            const float d = fp16_to_fp32(u16(b));
            for (int ib32 = 0; ib32 < GSQ_BLOCK_VALUES / 32; ++ib32) {
                const uint8_t *qs = b + 2 + 8 * ib32;   // the u16 codes, two bytes each
                const uint8_t sb = b[66 + ib32];
                const float db0 = d * (0.5f + (sb & 0xf)) * 0.25f;
                const float db1 = d * (0.5f + (sb >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint16_t code = u16(qs + 2 * l);
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(iq2xs_grid + (code & 511));
                    const uint8_t signs = ksigns_iq2xs[code >> 9];
                    const float db = l / 2 ? db1 : db0;
                    for (int j = 0; j < 8; ++j) {
                        y[l * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                y += 32;
            }
        } else if (type == GSQ_IQ1_S) {
            const float d = fp16_to_fp32(u16(b));
            const uint8_t *qs = b + 2;
            const uint16_t *qh = reinterpret_cast<const uint16_t *>(b + 34);
            for (int ib = 0; ib < GSQ_BLOCK_VALUES / 32; ++ib) {
                const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
                const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
                for (int l = 0; l < 4; ++l) {
                    const int8_t *grid = reinterpret_cast<const int8_t *>(
                        iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                    for (int j = 0; j < 8; ++j) {
                        y[l * 8 + j] = dl * (grid[j] + delta);
                    }
                }
                y += 32;
                qs += 4;
            }
        } else {   // GSQ_IQ1_M
            const uint16_t *sc = reinterpret_cast<const uint16_t *>(b + 48);   // the scales
            const uint16_t scale_bits =
                uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            const float d = fp16_to_fp32(scale_bits);
            const uint8_t *qs = b;
            const uint8_t *qh = b + 32;
            for (int ib = 0; ib < GSQ_BLOCK_VALUES / 32; ++ib) {
                const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
                const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                uint16_t idx[4];
                idx[0] = uint16_t(qs[0] | ((qh[0] << 8) & 0x700));
                idx[1] = uint16_t(qs[1] | ((qh[0] << 4) & 0x700));
                idx[2] = uint16_t(qs[2] | ((qh[1] << 8) & 0x700));
                idx[3] = uint16_t(qs[3] | ((qh[1] << 4) & 0x700));
                const float delta[4] = {qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                                        qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
                for (int l = 0; l < 2; ++l) {
                    const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                    const float dl = dl1;
                    for (int j = 0; j < 8; ++j) y[l * 8 + j] = dl * (grid[j] + delta[l]);
                }
                for (int l = 2; l < 4; ++l) {
                    const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                    const float dl = dl2;
                    for (int j = 0; j < 8; ++j) y[l * 8 + j] = dl * (grid[j] + delta[l]);
                }
                y += 32;
                qs += 4;
                qh += 2;
            }
        }
    }
}

}  // namespace xtier
