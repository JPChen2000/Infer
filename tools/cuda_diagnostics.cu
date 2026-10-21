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
