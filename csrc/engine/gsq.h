// The IQ quantization formats of the GSQ checkpoints (llama.cpp's IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS), decoded to FP32.
//
// The blocks hold 256 values each:
//   IQ2_XXS (16): fp16 d, u16 qs[32]                                    66 bytes
//   IQ2_XS  (17): fp16 d, u16 qs[32], u8 scales[8]                      74 bytes
//   IQ1_S   (19): fp16 d, u8 qs[32], u16 qh[16]                         50 bytes
//   IQ1_M   (29): u8 qs[32], u8 qh[16], u8 scales[8]                    56 bytes
//
// The values are the ones llama.cpp's dequantize_row_iq* produce (ggml/src/ggml-quants.c), so the engine computes
// the same numbers as the reference implementation.
#pragma once

#include <cstdint>

namespace xtier {

enum GsqType {
    GSQ_IQ2_XXS = 16,
    GSQ_IQ2_XS = 17,
    GSQ_IQ1_S = 19,
    GSQ_IQ1_M = 29,
};

constexpr int GSQ_BLOCK_VALUES = 256;   // the values of one block

// The bytes of a row of n values (n % 256 == 0), or 0 for a type the engine cannot read.
int gsq_row_bytes(int type, int n);

// One row of n values (n % 256 == 0) as FP32.
void gsq_dequant_row(int type, const uint8_t *src, float *dst, int n);

}  // namespace xtier
