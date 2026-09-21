// VNNI expert GEMM v4 arms (MosaicVM Track V).
//
// v3   : int8 qW [N, K], n-outer/m-inner, per-block hsum  (reference)
// v4a  : packed nibble reads (uint8 [N, K/2]) + on-the-fly pshufb LUT
//        unpack into the v3 structure — halves weight RAM traffic
// v4b  : tiled layout: qT [N/8, K/4, 32B] tiles = 8 outputs x 4 K per
//        dpbusd lane-parallel — no per-block horizontal sum
// v4ab : packed reads feeding tile construction (unpack amortized
//        across the m loop)
//
// All arms: exact MXIE-class integer arithmetic (see v3 header).

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline __m256i hsum8(__m256i d) {
    __m128i lo = _mm256_castsi256_si128(d);
    __m128i hi = _mm256_extracti128_si256(d, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    __m256i r = _mm256_setzero_si256();
    r = _mm256_insertf128_si256(r, s, 0);
    return r;   // sum broadcast? no — caller uses cvtsi128
}
static inline int hsum8i(__m256i d) {
    __m128i lo = _mm256_castsi256_si128(d);
    __m128i hi = _mm256_extracti128_si256(d, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// ---------------- v3 (reference, int8 qW) ----------------
void vnni_gemm_scaled2(const int8_t* __restrict__ mA,
                       const float* __restrict__ sA,
                       const int8_t* __restrict__ qW,
                       const float* __restrict__ sW,
                       float* __restrict__ y,
                       int M, int N, int K) {
    const int WB = K / 32;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int8_t* wrow = qW + (size_t)n * K;
        const float* swrow = sW + (size_t)n * WB;
        for (int m = 0; m < M; m++) {
            const int8_t* arow = mA + (size_t)m * K;
            const float* sarow = sA + (size_t)m * (K / 128);
            float yacc = 0.0f;
            for (int wb = 0; wb < WB; wb++) {
                const int k = wb * 32;
                __m256i a = _mm256_loadu_si256(
                    (const __m256i*)(arow + k));
                __m256i w = _mm256_loadu_si256(
                    (const __m256i*)(wrow + k));
                __m256i d = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), a, w);
                yacc += (float)hsum8i(d) * sarow[k / 128] * swrow[wb];
            }
            y[(size_t)m * N + n] = yacc * 0.5f;
        }
    }
}

// ---------------- v4a: packed nibble reads ----------------
// qP: uint8 [N, K/2]; nibble -> qW int8 via 16-byte LUT (pshufb).
static const signed char QLUT[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                     0, -1, -2, -3, -4, -6, -8, -12};

void vnni_gemm_packed(const int8_t* __restrict__ mA,
                      const float* __restrict__ sA,
                      const uint8_t* __restrict__ qP,   // [N, K/2]
                      const float* __restrict__ sW,
                      float* __restrict__ y,
                      int M, int N, int K) {
    const int WB = K / 32;
    const __m128i lut = _mm_loadu_si128((const __m128i*)QLUT);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint8_t* prow = qP + (size_t)n * (K / 2);
        const float* swrow = sW + (size_t)n * WB;
        for (int m = 0; m < M; m++) {
            const int8_t* arow = mA + (size_t)m * K;
            const float* sarow = sA + (size_t)m * (K / 128);
            float yacc = 0.0f;
            for (int wb = 0; wb < WB; wb++) {
                const int k = wb * 32;
                __m256i a = _mm256_loadu_si256(
                    (const __m256i*)(arow + k));
                // unpack 16 packed bytes -> 32 qW in K order
                __m128i pk = _mm_loadu_si128(
                    (const __m128i*)(prow + wb * 16));
                __m128i lo4 = _mm_and_si128(pk, _mm_set1_epi8(0x0F));
                __m128i hi4 = _mm_and_si128(
                    _mm_srli_epi16(pk, 4), _mm_set1_epi8(0x0F));
                __m128i qlo = _mm_shuffle_epi8(lut, lo4);
                __m128i qhi = _mm_shuffle_epi8(lut, hi4);
                __m128i w16a = _mm_unpacklo_epi8(qlo, qhi);
                __m128i w16b = _mm_unpackhi_epi8(qlo, qhi);
                __m256i w = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(w16a), w16b, 1);
                __m256i d = _mm256_dpbusd_epi32(
                    _mm256_setzero_si256(), a, w);
                yacc += (float)hsum8i(d) * sarow[k / 128] * swrow[wb];
            }
            y[(size_t)m * N + n] = yacc * 0.5f;
        }
    }
}

// ---------------- v4b: tiled 8-output layout ----------------
// qT: int8 [N/8, K/4, 32]: tile (nt, kt) = bytes j: qW[n0+j/4? ...
// layout: byte index b in [0,32): lane = b / 4, kk = b % 4 ->
//   qT[nt][kt][lane*4 + kk] = qW[n0 + lane][4*kt + kk]
// aT: uint8 [M, K/4, 32]: aT[m][kt][lane*4+kk] = mA[m][4*kt+kk]
//      (same 4 bytes replicated across the 8 lanes)
void vnni_gemm_tiled(const uint8_t* __restrict__ aT,
                     const float* __restrict__ sA,
                     const int8_t* __restrict__ qT,
                     const float* __restrict__ sW,   // [N, K/32]
                     float* __restrict__ y,
                     int M, int N, int K) {
    const int WB = K / 32;
    const int NT = N / 8;
    #pragma omp parallel for schedule(static)
    for (int nt = 0; nt < NT; nt++) {
        const int n0 = nt * 8;
        const int8_t* tbase = qT + (size_t)nt * (K / 4) * 32;
        for (int m = 0; m < M; m++) {
            const uint8_t* abase = aT + (size_t)m * (K / 4) * 32;
            const float* sarow = sA + (size_t)m * (K / 128);
            float yacc[8];
            for (int j = 0; j < 8; j++) yacc[j] = 0.0f;
            for (int wb = 0; wb < WB; wb++) {
                __m256i acc = _mm256_setzero_si256();
                for (int t = 0; t < 8; t++) {      // 8 x 4K tiles
                    const int kt = wb * 8 + t;
                    __m256i a = _mm256_loadu_si256(
                        (const __m256i*)(abase + (size_t)kt * 32));
                    __m256i w = _mm256_loadu_si256(
                        (const __m256i*)(tbase + (size_t)kt * 32));
                    acc = _mm256_dpbusd_epi32(acc, a, w);
                }
                int32_t tmp[8] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i*)tmp, acc);
                const float sa = sarow[wb / 4];
                for (int lane = 0; lane < 8; lane++)
                    yacc[lane] += (float)tmp[lane] * sa *
                        sW[(size_t)(n0 + lane) * WB + wb];
            }
            for (int lane = 0; lane < 8; lane++)
                y[(size_t)m * N + n0 + lane] = yacc[lane] * 0.5f;
        }
    }
}

// ---------------- v4ab: packed -> tile, unpack amortized over m ----
// qP: uint8 [N, K/2] packed (canonical). Tiles built per (nt, wb)
// into registers, reused across the m loop.
void vnni_gemm_packed_tiled(const uint8_t* __restrict__ aT,
                            const float* __restrict__ sA,
                            const uint8_t* __restrict__ qP,
                            const float* __restrict__ sW,
                            float* __restrict__ y,
                            int M, int N, int K) {
    const int WB = K / 32;
    const int NT = N / 8;
    const __m128i lut = _mm_loadu_si128((const __m128i*)QLUT);
    #pragma omp parallel for schedule(static)
    for (int nt = 0; nt < NT; nt++) {
        const int n0 = nt * 8;
        for (int wb = 0; wb < WB; wb++) {
            // build the 8 weight tiles for this (nt, wb) ONCE:
            // tile t covers K [wb*32 + 4t, +4): for each of 8 output
            // rows, 4 qW values = 2 packed bytes at
            // qP[n0+lane][ (wb*32+4t)/2 .. +1 ]
            __m256i wtiles[8];
            for (int t = 0; t < 8; t++) {
                // gather 8 rows x 2 bytes -> 32-byte tile with
                // lane-major layout: lane*4+kk
                int8_t tmp[32];
                for (int lane = 0; lane < 8; lane++) {
                    const uint8_t* p = qP +
                        (size_t)(n0 + lane) * (K / 2) +
                        (wb * 32 + 4 * t) / 2;
                    uint8_t b0 = p[0], b1 = p[1];
                    // k = 4t+0,1 from b0; 4t+2,3 from b1
                    tmp[lane * 4 + 0] = QLUT[b0 & 0x0F];
                    tmp[lane * 4 + 1] = QLUT[b0 >> 4];
                    tmp[lane * 4 + 2] = QLUT[b1 & 0x0F];
                    tmp[lane * 4 + 3] = QLUT[b1 >> 4];
                }
                wtiles[t] = _mm256_loadu_si256((const __m256i*)tmp);
            }
            const float* swl[8];
            for (int lane = 0; lane < 8; lane++)
                swl[lane] = sW + (size_t)(n0 + lane) * WB + wb;
            for (int m = 0; m < M; m++) {
                const uint8_t* abase = aT + (size_t)m * (K / 4) * 32;
                const float* sarow = sA + (size_t)m * (K / 128);
                __m256i acc = _mm256_setzero_si256();
                for (int t = 0; t < 8; t++) {
                    __m256i a = _mm256_loadu_si256(
                        (const __m256i*)(abase +
                            (size_t)(wb * 8 + t) * 32));
                    acc = _mm256_dpbusd_epi32(acc, a, wtiles[t]);
                }
                int32_t tmp2[8] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i*)tmp2, acc);
                const float sa = sarow[wb / 4];
                for (int lane = 0; lane < 8; lane++) {
                    const float v = (float)tmp2[lane] * sa *
                        swl[lane][0] * 0.5f;
                    if (wb == 0)
                        y[(size_t)m * N + n0 + lane] = v;
                    else
                        y[(size_t)m * N + n0 + lane] += v;
                }
            }
        }
    }
}

// pre-tiling helpers
void tile_weights(const int8_t* __restrict__ qW,  // [N, K]
                  int8_t* __restrict__ qT,        // [N/8, K/4, 32]
                  int N, int K) {
    for (int nt = 0; nt < N / 8; nt++)
        for (int kt = 0; kt < K / 4; kt++)
            for (int lane = 0; lane < 8; lane++)
                for (int kk = 0; kk < 4; kk++)
                    qT[((size_t)nt * (K / 4) + kt) * 32 + lane * 4 + kk]
                        = qW[(size_t)(nt * 8 + lane) * K + 4 * kt + kk];
}

void tile_acts(const int8_t* __restrict__ mA,     // [M, K]
               uint8_t* __restrict__ aT,          // [M, K/4, 32]
               int M, int K) {
    for (int m = 0; m < M; m++)
        for (int kt = 0; kt < K / 4; kt++)
            for (int lane = 0; lane < 8; lane++)
                for (int kk = 0; kk < 4; kk++)
                    aT[((size_t)m * (K / 4) + kt) * 32 + lane * 4 + kk]
                        = mA[(size_t)m * K + 4 * kt + kk];
}
