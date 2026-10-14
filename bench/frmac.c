// Throughput of y += g*x over the BLS12-381 scalar field Fr (Montgomery,
// 4x64-bit limbs, CIOS). This is the inner loop of RLNC coding (relay) and of
// decoding/recoding.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef unsigned __int128 u128;
static const uint64_t P[4] = {0xffffffff00000001ULL, 0x53bda402fffe5bfeULL,
                              0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL};
static const uint64_t INV = 0xfffffffeffffffffULL; // -p^{-1} mod 2^64
static inline void mont_mul(uint64_t r[4], const uint64_t a[4],
                            const uint64_t b[4]) {
  uint64_t t[6] = {0};
  for (int i = 0; i < 4; i++) {
    u128 c = 0;
    for (int j = 0; j < 4; j++) {
      c += (u128)a[j] * b[i] + t[j];
      t[j] = (uint64_t)c;
      c >>= 64;
    }
    c += t[4];
    t[4] = (uint64_t)c;
    t[5] = (uint64_t)(c >> 64);
    uint64_t m = t[0] * INV;
    c = (u128)m * P[0] + t[0];
    c >>= 64;
    for (int j = 1; j < 4; j++) {
      c += (u128)m * P[j] + t[j];
      t[j - 1] = (uint64_t)c;
      c >>= 64;
    }
    c += t[4];
    t[3] = (uint64_t)c;
    t[4] = t[5] + (uint64_t)(c >> 64);
  }
  // conditional subtraction
  uint64_t s[4];
  u128 br = 0;
  for (int j = 0; j < 4; j++) {
    u128 d = (u128)t[j] - P[j] - br;
    s[j] = (uint64_t)d;
    br = (d >> 64) & 1;
  }
  if (t[4] || !br) {
    for (int j = 0; j < 4; j++)
      r[j] = s[j];
  } else {
    for (int j = 0; j < 4; j++)
      r[j] = t[j];
  }
}
static inline void add_mod(uint64_t r[4], const uint64_t a[4],
                           const uint64_t b[4]) {
  u128 c = 0;
  uint64_t t[4];
  for (int j = 0; j < 4; j++) {
    c += (u128)a[j] + b[j];
    t[j] = (uint64_t)c;
    c >>= 64;
  }
  uint64_t s[4];
  u128 br = 0;
  for (int j = 0; j < 4; j++) {
    u128 d = (u128)t[j] - P[j] - br;
    s[j] = (uint64_t)d;
    br = (d >> 64) & 1;
  }
  if (c || !br) {
    for (int j = 0; j < 4; j++)
      r[j] = s[j];
  } else {
    for (int j = 0; j < 4; j++)
      r[j] = t[j];
  }
}
int main(int argc, char **argv) {
  long L = argc > 1 ? atol(argv[1]) : 32768;
  int K = 16; // one piece: kappa layers x L elements
  uint64_t (*x)[4] = malloc(sizeof(uint64_t[4]) * L * K),
           (*y)[4] = malloc(sizeof(uint64_t[4]) * L);
  uint64_t g[16][4];
  srand(1);
  for (long i = 0; i < L * K; i++)
    for (int j = 0; j < 4; j++)
      x[i][j] = ((uint64_t)rand() << 32 | rand()) &
                (j == 3 ? 0x3fffffffffffffffULL : ~0ULL);
  for (int i = 0; i < K; i++)
    for (int j = 0; j < 4; j++)
      g[i][j] = ((uint64_t)rand() << 32 | rand()) &
                (j == 3 ? 0x3fffffffffffffffULL : ~0ULL);
  int reps = 20;
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int r = 0; r < reps; r++) {
    for (long e = 0; e < L; e++) {
      uint64_t acc[4] = {0, 0, 0, 0}, t[4];
      for (int i = 0; i < K; i++) {
        mont_mul(t, g[i], x[i * L + e]);
        add_mod(acc, acc, t);
      }
      for (int j = 0; j < 4; j++)
        y[e][j] = acc[j];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &b);
  double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
  double macs = (double)reps * L * K;
  printf("{\"mac_per_s\": %.4g, \"ns_per_mac\": %.3f, \"piece_ms\": %.4f, "
         "\"chk\": %llu}\n",
         macs / s, s / macs * 1e9, s / reps * 1e3,
         (unsigned long long)y[L / 2][0]);
  return 0;
}
