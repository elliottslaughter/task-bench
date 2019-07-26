/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cmath>

#if (__AVX2__ == 1) || (__AVX__ == 1)
#include <immintrin.h>
#endif

#include <string.h>

#include "core.h"
#include "core_kernel.h"
#include "core_random.h"

#ifdef USE_BLAS_KERNEL
#include <mkl.h>
#endif

void execute_kernel_empty(const Kernel &kernel)
{
  // Do nothing...
}

long long execute_kernel_busy_wait(const Kernel &kernel)
{
  long long acc = 113;
  for (long iter = 0; iter < kernel.iterations; iter++) {
    acc = acc * 139 % 2147483647;
  }
  return acc;
}

void copy(char *scratch_ptr, size_t scratch_bytes)
{
  assert(scratch_bytes % 64 == 0);
  assert(reinterpret_cast<intptr_t>(scratch_ptr) % 32 == 0);

  size_t nb_m256 = scratch_bytes / 64;

  char *aligned_src = scratch_ptr;
  char *aligned_dst = scratch_ptr + scratch_bytes / 2;
  for (int i = 0; i < nb_m256; i++) {
    __m256d *dst_m256 = reinterpret_cast<__m256d *>(aligned_dst);
    *dst_m256 = _mm256_load_pd(reinterpret_cast<double *>(aligned_src));
    aligned_src += 32;
    aligned_dst += 32;
  }
}

void execute_kernel_memory(const Kernel &kernel,
                           char *scratch_ptr, size_t scratch_bytes,
                           long timestep)
{
  assert(kernel.samples == 1);
  for (long iter = 0; iter < kernel.iterations; ++iter) {
    copy(scratch_ptr, scratch_bytes);
  }
}

void execute_kernel_dgemm(const Kernel &kernel,
                           char *scratch_ptr, size_t scratch_bytes)
{
#ifdef USE_BLAS_KERNEL
  long long N = scratch_bytes / (3 * sizeof(double));
  int m, n, p;
  double alpha, beta;

  m = n = p = sqrt(N);
  alpha = 1.0; beta = 1.0;

  double *A = reinterpret_cast<double *>(scratch_ptr);
  double *B = reinterpret_cast<double *>(scratch_ptr + N * sizeof(double));
  double *C = reinterpret_cast<double *>(scratch_ptr + 2 * N * sizeof(double));

  for (long iter = 0; iter < kernel.iterations; iter++) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                m, n, p, alpha, A, p, B, n, beta, C, n);
  }
#else
  fprintf(stderr, "No BLAS is detected\n");
  fflush(stderr);
  abort();
#endif
}

void execute_kernel_daxpy(const Kernel &kernel,
                          char *scratch_large_ptr, size_t scratch_large_bytes,
                          long timestep)
{
#ifdef USE_BLAS_KERNEL
  for (long iter = 0; iter < kernel.iterations; iter++) {  
    size_t scratch_bytes = scratch_large_bytes / kernel.samples;
    int idx = (timestep * kernel.iterations + iter) % kernel.samples;
    char *scratch_ptr = scratch_large_ptr + idx * scratch_bytes;
  
    int N = scratch_bytes / (2 * sizeof(double));
    double alpha;

    alpha = 1.0;

    double *X = reinterpret_cast<double *>(scratch_ptr);
    double *Y = reinterpret_cast<double *>(scratch_ptr + N * sizeof(double));

    cblas_daxpy(N, alpha, X, 1, Y, 1);
  }
#else
  fprintf(stderr, "No BLAS is detected\n");
  fflush(stderr);
  abort();
#endif
}

double execute_kernel_compute(const Kernel &kernel)
{
#if __AVX2__ == 1
  __m256d A[16];
  
  for (int i = 0; i < 16; i++) {
    A[i] = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
  }
  
  for (long iter = 0; iter < kernel.iterations; iter++) {
    for (int i = 0; i < 16; i++) {
      A[i] = _mm256_fmadd_pd(A[i], A[i], A[i]);
    }
  }
#elif __AVX__ == 1
  __m256d A[16];
  
  for (int i = 0; i < 16; i++) {
    A[i] = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
  }
  
  for (long iter = 0; iter < kernel.iterations; iter++) {
    for (int i = 0; i < 16; i++) {
      A[i] = _mm256_mul_pd(A[i], A[i]);
      A[i] = _mm256_add_pd(A[i], A[i]);
    }
  }
#else
  double A[64];
  
  for (int i = 0; i < 64; i++) {
    A[i] = 1.2345;
  }
  
  for (long iter = 0; iter < kernel.iterations; iter++) {
    for (int i = 0; i < 64; i++) {
        A[i] = A[i] * A[i] + A[i];
    }
  } 
#endif
  double *C = (double *)A;
  double dot = 1.0;
  for (int i = 0; i < 64; i++) {
    dot *= C[i];
  }
  return dot;  
}

double execute_kernel_compute2(const Kernel &kernel)
{
  constexpr size_t N = 32;
  double A[N] = {0};
  double B[N] = {0};
  double C[N] = {0};

  for (size_t i = 0; i < N; ++i) {
    A[i] = 1.2345;
    B[i] = 1.010101;
  }

  for (long iter = 0; iter < kernel.iterations; iter++) {
    for (size_t i = 0; i < N; ++i) {
      C[i] = C[i] + (A[i] * B[i]);
    }
  }

  double sum = 0;
  for (size_t i = 0; i < N; ++i) {
    sum += C[i];
  }
  return sum;
}

void execute_kernel_io(const Kernel &kernel)
{
  assert(false);
}

long select_imbalance_iterations(const Kernel &kernel,
                                 long graph_index, long timestep, long point)
{
  long seed[3] = {graph_index, timestep, point};
  double value = random_uniform(&seed[0], sizeof(seed));

  long iterations = (long)round((1 + (value - 0.5)*kernel.imbalance) * kernel.iterations);
  assert(iterations >= 0);
  return iterations;
}

double execute_kernel_imbalance(const Kernel &kernel,
                                long graph_index, long timestep, long point)
{
  long iterations = select_imbalance_iterations(kernel, graph_index, timestep, point);
  Kernel k(kernel);
  k.iterations = iterations;
  // printf("iteration %ld\n", iterations);
  return execute_kernel_compute(k);
}
