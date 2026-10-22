// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <iostream>

__global__ void AddOne(float* value) { *value += 1.f; }

int main() {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::cerr << "CUDA enumeration failed: " << cudaGetErrorString(status) << "; devices=" << count << '\n';
        return 1;
    }
    bool failed = false;
    for (int device = 0; device < count; ++device) {
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            std::cerr << "GPU " << device << " context creation failed: " << cudaGetErrorString(status)
                      << ". Check free GPU memory with nvidia-smi.\n";
            failed = true; continue;
        }
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, device);
        size_t available = 0, total = 0;
        status = cudaMemGetInfo(&available, &total);
        std::cout << "GPU " << device << ": " << properties.name << ", sm_" << properties.major << properties.minor
                  << ", free=" << available / (1024 * 1024) << " MiB\n";
        float* value = nullptr;
        if (status == cudaSuccess) status = cudaMalloc(&value, sizeof(float));
        float host = 1.f;
        if (status == cudaSuccess) status = cudaMemcpy(value, &host, sizeof(float), cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            AddOne<<<1, 1>>>(value);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess) status = cudaMemcpy(&host, value, sizeof(float), cudaMemcpyDeviceToHost);
        if (value) cudaFree(value);
        if (status != cudaSuccess || host != 2.f) {
            std::cerr << "GPU " << device << " execution failed: " << cudaGetErrorString(status) << ", value=" << host << '\n';
            failed = true;
        } else std::cout << "GPU " << device << ": allocation, transfer and kernel execution passed\n";
    }
    return failed ? 1 : 0;
}
