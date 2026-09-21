// Source E4M3 x E2M1: exact qA=512*A, qW=2*W per-K32 integer dots.
// Activations encode u=qA+2^18 into three unsigned byte planes; subtract
// 2^18*sum(qW) BEFORE converting the reconstructed dot to FP32.
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

void vnni_source_gemm(const uint8_t *a, const float *sa,
                      const int8_t *w, const float *sw,
                      const int32_t *wsum, float *y, int M, int N, int K) {
    const int wb_count = K / 32;
    const size_t plane_size = (size_t)M * K * 8;
    #pragma omp parallel for schedule(static)
    for (int nt = 0; nt < N / 8; nt++) {
        for (int m = 0; m < M; m++) {
            float result[8] = {0};
            for (int wb = 0; wb < wb_count; wb++) {
                __m256i d0 = _mm256_setzero_si256();
                __m256i d1 = _mm256_setzero_si256();
                __m256i d2 = _mm256_setzero_si256();
                for (int t = 0; t < 8; t++) {
                    size_t kt = (size_t)wb * 8 + t;
                    size_t ai = ((size_t)m * (K / 4) + kt) * 32;
                    size_t wi = ((size_t)nt * (K / 4) + kt) * 32;
                    __m256i wt = _mm256_loadu_si256((const __m256i *)(w + wi));
                    d0 = _mm256_dpbusd_epi32(d0, _mm256_loadu_si256((const __m256i *)(a + ai)), wt);
                    d1 = _mm256_dpbusd_epi32(d1, _mm256_loadu_si256((const __m256i *)(a + plane_size + ai)), wt);
                    d2 = _mm256_dpbusd_epi32(d2, _mm256_loadu_si256((const __m256i *)(a + 2 * plane_size + ai)), wt);
                }
                __m256i dot = _mm256_add_epi32(d0, _mm256_slli_epi32(d1, 8));
                dot = _mm256_add_epi32(dot, _mm256_slli_epi32(d2, 16));
                int32_t sums[8];
                _mm256_storeu_si256((__m256i *)sums, dot);
                for (int lane = 0; lane < 8; lane++) {
                    size_t index = (size_t)(nt * 8 + lane) * wb_count + wb;
                    int32_t exact = sums[lane] - 262144 * wsum[index];
                    float partial = (float)exact * (1.0f / 1024.0f);
                    partial = partial * sa[(size_t)m * (K / 128) + wb / 4];
                    partial = partial * sw[index];
                    result[lane] = result[lane] + partial;
                }
            }
            for (int lane = 0; lane < 8; lane++)
                y[(size_t)m * N + nt * 8 + lane] = result[lane];
        }
    }
}
