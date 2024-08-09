#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

// declarations
extern "C" __global__ void cinn_kernel(const float* src, const float* src_1, const float* src_10, float* dst);
extern "C" __global__ void simplified_kernel(const float* src, const float* src_1, const float* src_10, float* dst);

// CUDA call wraper
#define CUDA_CALL(call) {                                    \
    const cudaError_t error = call;                           \
    if (error != cudaSuccess) {                               \
        std::cerr << "Error: " << cudaGetErrorString(error);  \
        throw std::runtime_error("CUDA API call failed");     \
    }                                                         \
}

int main() {
    try {
        // vector item size
        constexpr size_t srcSize = 1536 * 256 * 4;
        constexpr size_t dstSize = 1536*256 * 4;

        constexpr dim3 blocks(1536);
        constexpr dim3 threads(256);

        // DRAM buffer initialize
        std::vector<float> h_src(srcSize);
        std::vector<float> h_src_1(srcSize);
        std::vector<float> h_src_10(srcSize);
        std::vector<float> h_dst(dstSize);


        // random initialize
        std::uniform_real_distribution<float> dist(-__FLT_MAX__, __FLT_MAX__);
        std::mt19937_64 rng(std::random_device{}());
        #pragma loop_unroll(4)
        for(size_t i = 0; i < srcSize; ++i) {
          h_src[i] = dist(rng);
          h_src_1[i] = dist(rng);
          h_src_10[i] = dist(rng);
          h_dst[i] = dist(rng);
        }

        // device memory allocate
        float* d_src;
        float* d_src_1;
        float* d_src_10;
        float* d_dst;
        CUDA_CALL(cudaMalloc(&d_src, srcSize * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_src_1, srcSize * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_src_10, srcSize * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_dst, dstSize * sizeof(float)));

        // data xfer from host to device
        CUDA_CALL(cudaMemcpy(d_src, h_src.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_src_1, h_src_1.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_src_10, h_src_10.data(), srcSize * sizeof(float), cudaMemcpyHostToDevice));

        // summon cuda kernel
        //cinn_kernel<<<blocks, threads>>>(d_src, d_src_1, d_src_10, d_dst);
        simplified_kernel<<<blocks, threads>>>(d_src, d_src_1, d_src_10, d_dst);

        //  data xfer from device to host
        CUDA_CALL(cudaMemcpy(h_dst.data(), d_dst, dstSize * sizeof(float), cudaMemcpyDeviceToHost));

        // device mem free
        CUDA_CALL(cudaFree(d_src));
        CUDA_CALL(cudaFree(d_src_1));
        CUDA_CALL(cudaFree(d_src_10));
        CUDA_CALL(cudaFree(d_dst));


    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

extern "C" {

__global__
void __launch_bounds__(256) cinn_kernel(const float* __restrict__ src, const float* __restrict__ src_1, const float* __restrict__ src_10, float* __restrict__ dst)
{
  if (((int)blockIdx.x < 1536)) {
    for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
      if (((int)threadIdx.x < 256)) {
        float local_var = src_10[(((int)threadIdx.x & 127) + (((((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) / 96ll) * 1572864ll) + ((16384ll * (((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)) + (128ll * ((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) & 127)))))];
        float local_var_0 = src_1[(((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)];
        float local_var_1 = src[(((((int)threadIdx.x / 128ll) + ((8 * (int)blockIdx.x) + (2 * i_j_k_a_fused_0))) / 128ll) % 96ll)];
        dst[(((int)threadIdx.x & 127) + (((((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) / 96ll) * 1572864ll) + ((16384ll * (((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)) + (128ll * ((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) & 127)))))] = (local_var * (1.00000000f / (1.00000000f + exp(((-1.00000000f * (local_var_0 + local_var_1)) + 0.00000000f)))));
      };
    };
  };
}

__global__
void __launch_bounds__(256) simplified_kernel(const float* __restrict__ src, const float* __restrict__ src_1, const float* __restrict__ src_10, float* __restrict__ dst) {
  if (((int)blockIdx.x < 1536)) {
    for (int32_t i_j_k_a_fused_0 = 0; i_j_k_a_fused_0 < 4; i_j_k_a_fused_0 += 1) {
      if (((int)threadIdx.x < 256)) {
        float local_var = src_10[(((int)threadIdx.x & 127) + (((((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) / 96ll) * 1572864ll) + ((16384ll * (((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)) + (128ll * ((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) & 127)))))];
        float local_var_0 = src_1[(((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)];
        float local_var_1 = src[(((((int)threadIdx.x / 128ll) + ((8 * (int)blockIdx.x) + (2 * i_j_k_a_fused_0))) / 128ll) % 96ll)];
        dst[(((int)threadIdx.x & 127) + (((((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) / 96ll) * 1572864ll) + ((16384ll * (((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) / 128ll) % 96ll)) + (128ll * ((((int)threadIdx.x / 128ll) + ((8ll * (int)blockIdx.x) + (2ll * i_j_k_a_fused_0))) & 127)))))] = (local_var * (1.00000000f / (1.00000000f + exp(((-1.00000000f * (local_var_0 + local_var_1)) + 0.00000000f)))));
      };
    }
  }
}
}
