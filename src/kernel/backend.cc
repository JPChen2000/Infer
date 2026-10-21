#include "core/backend.h"
#ifdef FEATHER_WITH_CUDA
#include "src/kernel/cuda/runtime.h"
#endif

namespace feather {
#ifdef FEATHER_WITH_CUDA
namespace {
class CudaCapture final : public ExecutionCapture {
 public:
    CudaCapture(std::shared_ptr<kernel::cuda_detail::CudaExecutionContext> context, cudaGraphExec_t executable)
        : context_(std::move(context)), executable_(executable) {}
    ~CudaCapture() override {
        kernel::cuda_detail::ExecutionContextScope scope(context_);
        (void)kernel::cuda_detail::SynchronizeInferenceStream();
        (void)cudaGraphExecDestroy(executable_);
    }
    Status Launch() override {
        kernel::cuda_detail::ExecutionContextScope scope(context_);
        const auto stream = kernel::cuda_detail::InferenceStream();
        return stream && cudaGraphLaunch(executable_, stream) == cudaSuccess ? Status::Ok()
            : Status::Error(StatusCode::kExecutionFailed, "CUDA graph launch failed");
    }
 private:
    std::shared_ptr<kernel::cuda_detail::CudaExecutionContext> context_;
    cudaGraphExec_t executable_{nullptr};
};
class CudaBackendContext final : public BackendContext {
    using CudaScope = kernel::cuda_detail::ExecutionContextScope;
    struct ActiveScope final : Scope {
        explicit ActiveScope(const std::shared_ptr<kernel::cuda_detail::CudaExecutionContext>& context) : scope(context) {}
        CudaScope scope;
    };
 public:
    CudaBackendContext() : context_(kernel::cuda_detail::CreateExecutionContext()) {}
    BackendCapabilities Capabilities(DeviceType device) const override {
        return device == DeviceType::CUDA ? BackendCapabilities{true, false, false, false}
                                         : BackendContext::Capabilities(device);
    }
    std::unique_ptr<Scope> Enter() override { return std::make_unique<ActiveScope>(context_); }
    const std::shared_ptr<kernel::cuda_detail::CudaExecutionContext>& CudaContext() const override { return context_; }
    Status Capture(const std::function<Status()>& run, std::unique_ptr<ExecutionCapture>* result) override {
        if (!result || !run) return Status::Error(StatusCode::kInvalidArgument, "invalid capture destination or callback");
        auto scope = Enter();
        const auto stream = kernel::cuda_detail::InferenceStream();
        if (!stream || cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            (void)cudaGetLastError();
            return Status::Error(StatusCode::kUnsupported, "CUDA stream capture unavailable");
        }
        Status status;
        try { status = run(); } catch (const std::exception& error) {
            status = Status::Error(StatusCode::kExecutionFailed, error.what());
        }
        cudaGraph_t graph = nullptr;
        const auto ended = cudaStreamEndCapture(stream, &graph);
        if (!status.ok() || ended != cudaSuccess || !graph) {
            if (graph) (void)cudaGraphDestroy(graph);
            (void)cudaGetLastError();
            return status.ok() ? Status::Error(StatusCode::kUnsupported, "CUDA graph capture failed") : status;
        }
        cudaGraphExec_t executable = nullptr;
        const auto instantiated = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
        (void)cudaGraphDestroy(graph);
        if (instantiated != cudaSuccess || !executable) {
            if (executable) (void)cudaGraphExecDestroy(executable);
            (void)cudaGetLastError();
            return Status::Error(StatusCode::kUnsupported, "CUDA graph instantiation failed");
        }
        *result = std::make_unique<CudaCapture>(context_, executable);
        return Status::Ok();
    }
    Status Synchronize() override {
        auto scope = Enter();
        return Result(kernel::cuda_detail::SynchronizeInferenceStream(), "CUDA synchronization failed");
    }
    void SetWorkspaceLimit(size_t bytes) override {
        auto scope = Enter(); kernel::cuda_detail::SetWorkspaceLimit(bytes);
    }
    Status ReadHost(Tensor& tensor) override {
        auto scope = Enter();
        size_t bytes = 0;
        if (!CheckedTensorBytes(tensor.dims().data(), tensor.data_type(), &bytes)) bytes = tensor.memory_size();
        return Result(kernel::cuda_detail::SyncTensorToHostIfNeeded(&tensor, bytes, tensor.raw_data()), "CUDA host read failed");
    }
    void HostWritten(Tensor& tensor) override {
        auto scope = Enter(); kernel::cuda_detail::InvalidateTensorDevice(&tensor);
    }
    void MarkPersistent(const Tensor& tensor, bool persistent) override {
        auto scope = Enter(); kernel::cuda_detail::MarkTensorDevicePersistent(&tensor, persistent);
    }
    bool IsPersistent(const Tensor& tensor) const override {
        CudaScope scope(context_); return kernel::cuda_detail::IsTensorDevicePersistent(&tensor);
    }
    void Release(const Tensor& tensor) override {
        auto scope = Enter(); kernel::cuda_detail::ReleaseTensorDevice(&tensor);
    }
    std::shared_ptr<const Storage> DeviceStorage(const Tensor& tensor) const override {
        CudaScope scope(context_); return kernel::cuda_detail::GetTensorDeviceStorage(&tensor);
    }
    void Clear() override { auto scope = Enter(); kernel::cuda_detail::ClearTensorCache(); }
    Status SwapState(Tensor& input, Tensor& output) override {
        auto scope = Enter(); return Result(kernel::cuda_detail::SwapTensorDeviceStorage(&input, &output), "CUDA state swap failed");
    }
    Status AppendState(Tensor& input, const Tensor& output, int axis) override {
        auto scope = Enter(); return Result(kernel::cuda_detail::AppendTensorStateOnDevice(&input, &output, axis), "CUDA state append failed");
    }
 private:
    static Status Result(int code, const char* message) {
        return code == 0 ? Status::Ok() : Status::Error(StatusCode::kExecutionFailed,
            std::string(message) + ": " + kernel::cuda_detail::CudaLastErrorMessage());
    }
    std::shared_ptr<kernel::cuda_detail::CudaExecutionContext> context_;
};
}
std::unique_ptr<BackendContext> CreateBackendContext() { return std::make_unique<CudaBackendContext>(); }
#else
std::unique_ptr<BackendContext> CreateBackendContext() { return std::make_unique<BackendContext>(); }
#endif
}  // namespace feather
