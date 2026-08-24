#include "src/kernel/qwen_gated_delta.h"

#include <memory>

#include "src/kernel/cuda/kernel_io.cuh"
#include "util/timer.h"

namespace feather {
namespace kernel {
namespace {

bool ValidateState(const operators::QwenGatedDeltaStateParam* param, int64_t* heads, int64_t* key, int64_t* value) {
    if (param == nullptr || heads == nullptr || key == nullptr || value == nullptr || param->state == nullptr ||
        param->k == nullptr || param->v == nullptr || param->beta == nullptr || param->decay == nullptr ||
        param->out == nullptr || !param->state->IsInitialized() || !param->k->IsInitialized() ||
        !param->v->IsInitialized() || !param->beta->IsInitialized() || !param->decay->IsInitialized() ||
        !param->out->IsInitialized() || param->state->data_type() != DataType::FP32 ||
        param->k->data_type() != DataType::FP32 || param->v->data_type() != DataType::FP32 ||
        param->beta->data_type() != DataType::FP32 || param->decay->data_type() != DataType::FP32) {
        return false;
    }
    const auto& dims = param->state->dims();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 ||
        param->out->numel() != param->state->numel() || param->k->numel() != dims[1] * dims[2] ||
        param->v->numel() != dims[1] * dims[3] || param->beta->numel() != dims[1] ||
        param->decay->numel() != dims[1]) {
        return false;
    }
    *heads = dims[1];
    *key = dims[2];
    *value = dims[3];
    return true;
}

bool ValidateOutput(const operators::QwenGatedDeltaOutputParam* param, int64_t* heads, int64_t* key,
                    int64_t* value) {
    if (param == nullptr || heads == nullptr || key == nullptr || value == nullptr || param->state == nullptr ||
        param->q == nullptr || param->out == nullptr || !param->state->IsInitialized() || !param->q->IsInitialized() ||
        !param->out->IsInitialized() || param->state->data_type() != DataType::FP32 ||
        param->q->data_type() != DataType::FP32) {
        return false;
    }
    const auto& dims = param->state->dims();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 ||
        param->q->numel() != dims[1] * dims[2] || param->out->numel() != dims[1] * dims[3]) {
        return false;
    }
    *heads = dims[1];
    *key = dims[2];
    *value = dims[3];
    return true;
}

bool ValidateCombined(const operators::QwenGatedDeltaParam* param, int64_t* heads, int64_t* key, int64_t* value) {
    if (param == nullptr || heads == nullptr || key == nullptr || value == nullptr || param->state == nullptr ||
        param->k == nullptr || param->v == nullptr || param->beta == nullptr || param->decay == nullptr ||
        param->q == nullptr || param->next_state == nullptr || param->out == nullptr ||
        !param->state->IsInitialized() || !param->k->IsInitialized() || !param->v->IsInitialized() ||
        !param->beta->IsInitialized() || !param->decay->IsInitialized() || !param->q->IsInitialized() ||
        !param->next_state->IsInitialized() || !param->out->IsInitialized() || param->state->data_type() != DataType::FP32 ||
        param->k->data_type() != DataType::FP32 || param->v->data_type() != DataType::FP32 ||
        param->beta->data_type() != DataType::FP32 || param->decay->data_type() != DataType::FP32 ||
        param->q->data_type() != DataType::FP32) {
        return false;
    }
    const auto& dims = param->state->dims();
    if (dims.size() != 4 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0 ||
        param->next_state->numel() != param->state->numel() || param->out->numel() != dims[1] * dims[3] ||
        param->k->numel() != dims[1] * dims[2] || param->v->numel() != dims[1] * dims[3] ||
        param->beta->numel() != dims[1] || param->decay->numel() != dims[1] || param->q->numel() != dims[1] * dims[2]) {
        return false;
    }
    *heads = dims[1];
    *key = dims[2];
    *value = dims[3];
    return true;
}

__global__ void QwenGatedDeltaStateKernelCuda(const float* state, const float* k, const float* v, const float* beta,
                                              const float* decay, float* out, int64_t heads, int64_t key,
                                              int64_t value) {
    const int64_t head = static_cast<int64_t>(blockIdx.x);
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    if (head >= heads) {
        return;
    }
    for (int64_t j = lane; j < value; j += blockDim.x) {
        float kv = 0.0f;
        for (int64_t i = 0; i < key; ++i) {
            kv += state[(head * key + i) * value + j] * (decay[head] * k[head * key + i]);
        }
        for (int64_t i = 0; i < key; ++i) {
            const int64_t index = (head * key + i) * value + j;
            out[index] = state[index] * decay[head] + k[head * key + i] * (v[head * value + j] - kv) * beta[head];
        }
    }
}

__global__ void QwenGatedDeltaOutputKernelCuda(const float* state, const float* q, float* out, int64_t heads,
                                               int64_t key, int64_t value) {
    const int64_t head = static_cast<int64_t>(blockIdx.x);
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    if (head >= heads) {
        return;
    }
    for (int64_t j = lane; j < value; j += blockDim.x) {
        float sum = 0.0f;
        for (int64_t i = 0; i < key; ++i) {
            sum += state[(head * key + i) * value + j] * q[head * key + i];
        }
        out[head * value + j] = sum;
    }
}

__global__ void QwenGatedDeltaCombinedKernelCuda(const float* state, const float* k, const float* v,
                                                 const float* beta, const float* decay, const float* q,
                                                 float* next_state, float* out, int64_t heads, int64_t key,
                                                 int64_t value) {
    extern __shared__ float kv[];
    const int64_t head = static_cast<int64_t>(blockIdx.x);
    const int64_t lane = static_cast<int64_t>(threadIdx.x);
    if (head >= heads) {
        return;
    }
    for (int64_t j = lane; j < value; j += blockDim.x) {
        float sum = 0.0f;
        for (int64_t i = 0; i < key; ++i) {
            sum += state[(head * key + i) * value + j] * (decay[head] * k[head * key + i]);
        }
        kv[j] = sum;
        out[head * value + j] = 0.0f;
    }
    __syncthreads();
    for (int64_t j = lane; j < value; j += blockDim.x) {
        float output = 0.0f;
        for (int64_t i = 0; i < key; ++i) {
            const int64_t index = (head * key + i) * value + j;
            const float updated = state[index] * decay[head] +
                                  k[head * key + i] * (v[head * value + j] - kv[j]) * beta[head];
            next_state[index] = updated;
            output += updated * q[head * key + i];
        }
        out[head * value + j] = output;
    }
}

int RunState(operators::QwenGatedDeltaStateParam* param, int64_t heads, int64_t key, int64_t value) {
    cuda_detail::DeviceBuffer<float> state;
    cuda_detail::DeviceBuffer<float> k;
    cuda_detail::DeviceBuffer<float> v;
    cuda_detail::DeviceBuffer<float> beta;
    cuda_detail::DeviceBuffer<float> decay;
    cuda_detail::DeviceBuffer<float> out;
    if (cuda_detail::CopyTensorToDevice(param->state.get(), &state) != 0 ||
        cuda_detail::CopyTensorToDevice(param->k.get(), &k) != 0 ||
        cuda_detail::CopyTensorToDevice(param->v.get(), &v) != 0 ||
        cuda_detail::CopyTensorToDevice(param->beta.get(), &beta) != 0 ||
        cuda_detail::CopyTensorToDevice(param->decay.get(), &decay) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    QwenGatedDeltaStateKernelCuda<<<static_cast<int>(heads), cuda_detail::kCudaThreads, 0,
                                    cuda_detail::InferenceStream()>>>(state.get(), k.get(), v.get(), beta.get(),
                                                                        decay.get(), out.get(), heads, key, value);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

int RunOutput(operators::QwenGatedDeltaOutputParam* param, int64_t heads, int64_t key, int64_t value) {
    cuda_detail::DeviceBuffer<float> state;
    cuda_detail::DeviceBuffer<float> q;
    cuda_detail::DeviceBuffer<float> out;
    if (cuda_detail::CopyTensorToDevice(param->state.get(), &state) != 0 ||
        cuda_detail::CopyTensorToDevice(param->q.get(), &q) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    QwenGatedDeltaOutputKernelCuda<<<static_cast<int>(heads), cuda_detail::kCudaThreads, 0,
                                     cuda_detail::InferenceStream()>>>(state.get(), q.get(), out.get(), heads, key,
                                                                         value);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

int RunCombined(operators::QwenGatedDeltaParam* param, int64_t heads, int64_t key, int64_t value) {
    cuda_detail::DeviceBuffer<float> state;
    cuda_detail::DeviceBuffer<float> k;
    cuda_detail::DeviceBuffer<float> v;
    cuda_detail::DeviceBuffer<float> beta;
    cuda_detail::DeviceBuffer<float> decay;
    cuda_detail::DeviceBuffer<float> q;
    cuda_detail::DeviceBuffer<float> next_state;
    cuda_detail::DeviceBuffer<float> out;
    if (cuda_detail::CopyTensorToDevice(param->state.get(), &state) != 0 ||
        cuda_detail::CopyTensorToDevice(param->k.get(), &k) != 0 ||
        cuda_detail::CopyTensorToDevice(param->v.get(), &v) != 0 ||
        cuda_detail::CopyTensorToDevice(param->beta.get(), &beta) != 0 ||
        cuda_detail::CopyTensorToDevice(param->decay.get(), &decay) != 0 ||
        cuda_detail::CopyTensorToDevice(param->q.get(), &q) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->next_state.get(), &next_state) != 0 ||
        cuda_detail::AllocateTensorOnDevice(param->out.get(), &out) != 0) {
        return -1;
    }
    const size_t shared_bytes = static_cast<size_t>(value) * sizeof(float);
    QwenGatedDeltaCombinedKernelCuda<<<static_cast<int>(heads), cuda_detail::kCudaThreads, shared_bytes,
                                       cuda_detail::InferenceStream()>>>(
        state.get(), k.get(), v.get(), beta.get(), decay.get(), q.get(), next_state.get(), out.get(), heads, key,
        value);
    if (cuda_detail::CudaCheck(cudaGetLastError()) != 0) {
        return -1;
    }
    if (cuda_detail::CopyDeviceToTensor(&next_state, param->next_state.get()) != 0) {
        return -1;
    }
    return cuda_detail::CopyDeviceToTensor(&out, param->out.get());
}

bool g_cuda_qwen_gated_delta_kernels_registered = []() {
    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "QwenGatedDeltaState", []() {
        return std::make_unique<QwenGatedDeltaStateKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "QwenGatedDeltaOutput", []() {
        return std::make_unique<QwenGatedDeltaOutputKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    dispatcher.registerKernel(DeviceType::CUDA, DataType::FP32, "QwenGatedDelta", []() {
        return std::make_unique<QwenGatedDeltaKernel<DeviceType::CUDA, DataType::FP32>>();
    });
    return true;
}();

}  // namespace

template <>
int32_t QwenGatedDeltaStateKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::QwenGatedDeltaState::FP32");
    auto* param = static_cast<operators::QwenGatedDeltaStateParam*>(param_);
    int64_t heads = 0;
    int64_t key = 0;
    int64_t value = 0;
    return ValidateState(param, &heads, &key, &value) ? RunState(param, heads, key, value) : -1;
}

template <>
int32_t QwenGatedDeltaOutputKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::QwenGatedDeltaOutput::FP32");
    auto* param = static_cast<operators::QwenGatedDeltaOutputParam*>(param_);
    int64_t heads = 0;
    int64_t key = 0;
    int64_t value = 0;
    return ValidateOutput(param, &heads, &key, &value) ? RunOutput(param, heads, key, value) : -1;
}

template <>
int32_t QwenGatedDeltaKernel<DeviceType::CUDA, DataType::FP32>::compute() {
    AutoTimer timer("CUDA::QwenGatedDelta::FP32");
    auto* param = static_cast<operators::QwenGatedDeltaParam*>(param_);
    int64_t heads = 0;
    int64_t key = 0;
    int64_t value = 0;
    return ValidateCombined(param, &heads, &key, &value) ? RunCombined(param, heads, key, value) : -1;
}

void EnsureCudaQwenGatedDeltaKernelsRegistered() { (void)g_cuda_qwen_gated_delta_kernels_registered; }

}  // namespace kernel
}  // namespace feather
