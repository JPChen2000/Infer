// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#include "src/kernel/qwen_gated_delta_sequence.h"

#include <limits>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {
__global__ void QwenGatedDeltaSequenceCuda(const float* initial_state, const float* k, const float* v,
                                           const float* beta, const float* decay, const float* q,
                                           float* next_state, float* out, int64_t sequence, int64_t heads,
                                           int64_t key, int64_t value) {
    const int64_t h = blockIdx.x;
    // Each lane owns complete value columns through all time steps. Columns
    // never communicate, so time advancement requires no block or host barrier.
    for (int64_t j = threadIdx.x; j < value; j += blockDim.x) {
        for (int64_t t = 0; t < sequence; ++t) {
            const int64_t token_head = t * heads + h;
            const float* state = t == 0 ? initial_state : next_state;
            float kv = 0.0f;
            for (int64_t i = 0; i < key; ++i) {
                kv += state[(h * key + i) * value + j] * (decay[token_head] * k[token_head * key + i]);
            }
            float output = 0.0f;
            for (int64_t i = 0; i < key; ++i) {
                const int64_t index = (h * key + i) * value + j;
                const float updated = state[index] * decay[token_head] +
                    k[token_head * key + i] * (v[token_head * value + j] - kv) * beta[token_head];
                next_state[index] = updated;
                output += updated * q[token_head * key + i];
            }
            out[token_head * value + j] = output;
        }
    }
}
} // namespace

template <>
int32_t QwenGatedDeltaSequenceKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::QwenGatedDeltaSequence::FP32");
    auto* p = ParamAs<operators::QwenGatedDeltaSequenceParam>();
    if (!operators::qwen_sequence_detail::Validate(p, true) ||
        p->state->dims()[1] > std::numeric_limits<int>::max()) return -1;
    const int64_t heads = p->state->dims()[1];
    cuda_detail::DeviceBuffer<float> state, k, v, beta, decay, q, next_state, out;
    if (cuda_detail::CopyTensorToDevice(p->state.get(), &state) != 0 ||
        cuda_detail::CopyTensorToDevice(p->k.get(), &k) != 0 ||
        cuda_detail::CopyTensorToDevice(p->v.get(), &v) != 0 ||
        cuda_detail::CopyTensorToDevice(p->beta.get(), &beta) != 0 ||
        cuda_detail::CopyTensorToDevice(p->decay.get(), &decay) != 0 ||
        cuda_detail::CopyTensorToDevice(p->q.get(), &q) != 0 ||
        cuda_detail::AllocateTensorOnDevice(p->next_state.get(), &next_state) != 0 ||
        cuda_detail::AllocateTensorOnDevice(p->out.get(), &out) != 0) return -1;
    QwenGatedDeltaSequenceCuda<<<static_cast<int>(heads), cuda_detail::kCudaThreads, 0,
                                cuda_detail::InferenceStream()>>>(
        state.get(), k.get(), v.get(), beta.get(), decay.get(), q.get(), next_state.get(), out.get(),
        p->k->dims()[1], heads, p->state->dims()[2], p->state->dims()[3]);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) return -1;
    // Runtime execution defers host synchronization, retaining both outputs in
    // the shared CUDA context for following operators and captured execution.
    if (cuda_detail::CopyDeviceToTensor(&next_state, p->next_state.get()) != 0) return -1;
    return cuda_detail::CopyDeviceToTensor(&out, p->out.get());
}

void EnsureCudaQwenGatedDeltaSequenceKernelsRegistered() {
    static const bool registered = [] {
        KernelDispatcher::instance().registerKernel(DeviceType::CUDA, DataType::FP32,
            "QwenGatedDeltaSequence", [] {
                return std::make_unique<QwenGatedDeltaSequenceKernel<DeviceType::CUDA, DataType::FP32>>();
            });
        return true;
    }();
    (void)registered;
}
} // namespace kernel
} // namespace feather
