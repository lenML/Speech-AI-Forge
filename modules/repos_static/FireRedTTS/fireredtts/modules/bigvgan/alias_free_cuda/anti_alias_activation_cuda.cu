/* coding=utf-8
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "type_shim.h"
#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <c10/macros/Macros.h>

namespace {

    /*
template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 1>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 4>(c10::BFloat16 *dst, const c10::BFloat16 *src) { *((float2*) dst) = *((float2*) src); }

template <>
__device__ __inline__ void copy_vector<c10::Half, 1>(c10::Half *dst, const c10::Half *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<c10::Half, 4>(c10::Half *dst, const c10::Half *src) { *((float2*) dst) = *((float2*) src); }

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) { *dst = *src; }

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {*((half2*) dst) = *((half2*) src); }

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}
*/

template <typename input_t, typename output_t, typename acc_t>
__global__ void anti_alias_activation_forward(
    output_t *dst,
    const input_t *src,
    const input_t *ftr,
    const input_t *alpha,
    const input_t *beta,
    int batch_size,
    int channels,
    int seq_len)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    constexpr int ELEMENTS_PER_LDG_STG = 1; //(WARP_ITERATIONS < 4) ? 1 : 4;
    constexpr int BUFFER_SIZE = 32;
    constexpr int FILTER_SIZE = 12;
    constexpr int HALF_FILTER_SIZE = 6;
    constexpr int REPLICATION_PAD = 5; // 5 on each side

    // blockDim/threadIdx = (128, 1, 1)
    // gridDim/blockIdx = (seq_blocks, channels, batches)
    int block_offset = (blockIdx.x * 128 * BUFFER_SIZE + seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    int local_offset = threadIdx.x * BUFFER_SIZE;
    int seq_offset = blockIdx.x * 128 * BUFFER_SIZE + local_offset;


    //int intermediate_seq_len = seq_len * 2 - 1 + 4 * REPLICATION_PAD;
    //int intermediate_block_offset = (blockIdx.x * 128 * BUFFER_SIZE * 2 + intermediate_seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    //int intermediate_local_offset = threadIdx.x * BUFFER_SIZE * 2;

    int output_seq_len = seq_len * 2 ; //
    int output_block_offset = (blockIdx.x * 128 * BUFFER_SIZE * 2 + output_seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    int output_local_offset = threadIdx.x * BUFFER_SIZE * 2;
    int output_seq_offset = blockIdx.x * 128 * BUFFER_SIZE *2 + output_local_offset;
    // get values needed for replication padding before moving pointer
    const input_t *right_most_pntr = src + (seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
    input_t seq_left_most_value = right_most_pntr[0];
    input_t seq_right_most_value = right_most_pntr[seq_len - 1];

    src += block_offset + local_offset;
    dst += output_block_offset + output_local_offset  ;
    alpha = alpha + blockIdx.y;
    input_t alpha_val = expf(alpha[0]);
    beta = beta + blockIdx.y;
    input_t beta_val = expf(beta[0]);
    // load data from global memory
    input_t elements[2*FILTER_SIZE+2*BUFFER_SIZE] = {0};
    input_t intermediates[2*FILTER_SIZE+2*BUFFER_SIZE] = {0};
    //output_t output[2*BUFFER_SIZE];
    input_t filter[FILTER_SIZE];
    //input_t temp_data[ELEMENTS_PER_LDG_STG];
    //uint8_t temp_mask[ELEMENTS_PER_LDG_STG];

    #pragma unroll
    for (int it = 0; it < FILTER_SIZE; it+=1) {
        filter[it] = ftr[it];
    }


    #pragma unroll
    for (int it = -HALF_FILTER_SIZE;  it < BUFFER_SIZE + HALF_FILTER_SIZE ;  it+=1) {
        int element_index = seq_offset + it;
	if ((element_index < 0) && (element_index >= -REPLICATION_PAD)) {
	    elements[2*(HALF_FILTER_SIZE+it)] = 2*seq_left_most_value;
	}
	if ((element_index >= seq_len) && (element_index < seq_len + REPLICATION_PAD)) {
	    elements[2*(HALF_FILTER_SIZE+it)] = 2*seq_right_most_value;
	}
        if ((element_index >= 0) && (element_index < seq_len)) {
	  elements[2*(HALF_FILTER_SIZE+it)] = 2*src[it];
        }
    }



    // apply filter
    #pragma unroll
    for (int it = 0;  it < (2 * BUFFER_SIZE + 2*FILTER_SIZE);  it+=1) {
        input_t acc = 0.0;

	int element_index = output_seq_offset + it; // index for output
	#pragma unroll
        for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx+=1){
	  if ((element_index + f_idx) >= 0){
            acc += filter[f_idx] * elements[it+f_idx];
	  }
	}
        intermediates[it] = acc;
    }

    double no_div_by_zero = 0.000000001;
    #pragma unroll
    for (int it = 0; it < 12 + 2 * BUFFER_SIZE; it++) {
        intermediates[it] += (1.0/(beta_val + no_div_by_zero)) *  sinf(intermediates[it] * alpha_val) * sinf(intermediates[it] * alpha_val);
    }


    // now copy to output
    #pragma unroll
    for (int it = 0; it < 2*BUFFER_SIZE; it+=1){
      int element_index = output_seq_offset + it;
      if (element_index < output_seq_len) {
	dst[it]  = intermediates[it+6];
      }
    }



    // for (int it = 0;  it < BUFFER_SIZE;  it+=ELEMENTS_PER_LDG_STG) {
    //     int element_index = seq_offset + it;
    //     if (element_index < seq_len) {
    //         dst[it] = output[it];
    //     }
    // }


    // // Upsample convolution
    // for (int it = 0;  it < 2 * BUFFER_SIZE + 12;  it+=1) {
    //     input_t acc = 0.0;

    //     for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx+=1){
    //         acc += filter[f_idx] * elements[it+f_idx];
    //     }
    //     intermediates[it] = acc;
    // }

    // // correct the corners of intermediates
    // if (seq_offset == 0) {
    //     for (int it = 0; it < 6; it+=1)
    //         intermediates[it] = 0;
    // }

    // if (seq_offset + 32 >= seq_len) {
    //     int offset = seq_len % 32 == 0 ? 32 : seq_len % 32;

    //     for (int it = 0; it < 6; it++) {
    //         intermediates[6+2*offset+it] = 0;
    //     }
    // }




    // for (int it = 0;  it < BUFFER_SIZE;  it+=ELEMENTS_PER_LDG_STG) {
    //     int element_index = seq_offset + it;
    //     if (element_index < seq_len) {
    //         dst[it] = output[it];
    //     }
    // }
}

template<typename input_t, typename output_t, typename acc_t>
void dispatch_anti_alias_activation_forward(
    output_t *dst,
    const input_t *src,
    const input_t *ftr,
    const input_t *alpha,
    const input_t *beta,
    int batch_size,
    int channels,
    int seq_len)
{
    if (seq_len == 0) {
        return;
    } else {
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        constexpr int seq_len_per_block = 4096;
        int blocks_per_seq_len = (seq_len + seq_len_per_block - 1) / seq_len_per_block;
        dim3 blocks(blocks_per_seq_len, channels, batch_size);
        dim3 threads(threads_per_block, 1, 1);

        anti_alias_activation_forward<input_t, output_t, acc_t>
	  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, ftr, alpha, beta, batch_size, channels, seq_len);
    }
}
}

namespace anti_alias_activation {

  torch::Tensor fwd_cuda(torch::Tensor const& input, torch::Tensor const& filter, torch::Tensor const& alpha, torch::Tensor const& beta)
{
  // input is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
  const int batches = input.size(0);
  const int channels = input.size(1);
  const int seq_len = input.size(2);

  // Output
  auto act_options = input.options().requires_grad(false);
  int output_seq_len = seq_len*2; // we'll be dilating between each element by interspersing with zeros

  torch::Tensor anti_alias_activation_results =
      torch::empty({batches, channels, output_seq_len}, act_options);

  // Softmax Intermediate Result Ptr
  void* input_ptr = static_cast<void*>(input.data_ptr());
  void* filter_ptr = static_cast<void*>(filter.data_ptr());
  void* alpha_ptr = static_cast<void*>(alpha.data_ptr());
  void* beta_ptr = static_cast<void*>(beta.data_ptr());
  void* anti_alias_activation_results_ptr = static_cast<void*>(anti_alias_activation_results.data_ptr());

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(),
      "dispatch anti alias activation_forward",
      dispatch_anti_alias_activation_forward<scalar_t, scalar_t, float>(
        reinterpret_cast<scalar_t*>(anti_alias_activation_results_ptr),
	    reinterpret_cast<const scalar_t*>(input_ptr),
        reinterpret_cast<const scalar_t*>(filter_ptr),
        reinterpret_cast<const scalar_t*>(alpha_ptr),
	reinterpret_cast<const scalar_t*>(beta_ptr),
	    batches,
        channels,
        seq_len);
      );
  return anti_alias_activation_results;
}
}
