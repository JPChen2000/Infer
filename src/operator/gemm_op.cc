#include "src/operator/gemm_op.h"

#include <functional>
#include <numeric>
#include <utility>

#include "core/operator_registry.h"
#include "util/types.h"

namespace feather {
namespace operators {

namespace {

int32_t GetIntAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes, const std::string& key,
                        int32_t default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<int64_t>(&it->second); value != nullptr) {
        return static_cast<int32_t>(*value);
    }
    return default_value;
}

float GetFloatAttribute(const std::unordered_map<std::string, model::AttributeValue>& attributes,
                        const std::string& key, float default_value) {
    auto it = attributes.find(key);
    if (it == attributes.end()) {
        return default_value;
    }
    if (auto value = std::get_if<float>(&it->second); value != nullptr) {
        return *value;
    }
    return default_value;
}

bool IsVectorBias(const Tensor* bias, int64_t n) {
    if (bias == nullptr || !bias->IsInitialized() || bias->dims().empty() ||
        bias->dims()[bias->dims().size() - 1] != n) {
        return false;
    }
    for (size_t index = 0; index + 1 < bias->dims().size(); ++index) {
        if (bias->dims()[index] != 1) {
            return false;
        }
    }
    return bias->numel() == n;
}

std::shared_ptr<OpBase> BuildGemmOp(const model::NodeDesc& node, OperatorRegistry::TensorMap& tensors, const OperatorRegistry::BuildContext& context) {
    if (node.inputs.size() < 2 || node.outputs.size() != 1) {
        return nullptr;
    }

    GemmParam param;
    param.a = tensors[node.inputs[0]];
    param.b = tensors[node.inputs[1]];
    param.bias = node.inputs.size() > 2 ? tensors[node.inputs[2]] : nullptr;
    param.out = tensors[node.outputs[0]];
    param.alpha = GetFloatAttribute(node.attributes, "alpha", 1.0f);
    param.beta = GetFloatAttribute(node.attributes, "beta", 1.0f);
    param.trans_a = GetIntAttribute(node.attributes, "transA", 0) != 0;
    param.trans_b = GetIntAttribute(node.attributes, "transB", 0) != 0;
    if (param.a == nullptr || param.b == nullptr || param.out == nullptr) {
        return nullptr;
    }

    auto op = std::make_shared<GemmOp>(node.name.empty() ? "gemm" : node.name, param);
    if (op->CheckShape() != 0 || op->InferOutputShapes() != 0) {
        return nullptr;
    }
    kernel::EnsureGemmKernelsRegistered();
    auto kernel = CreateKernelForTensor(context.device, "Gemm", {param.a, param.b, param.bias, param.out}, DataType::FP32);
    if (kernel == nullptr) {
        return nullptr;
    }
    op->AttachKernel(std::move(kernel));
    return op;
}

bool g_gemm_op_registered = []() {
    OperatorRegistry::instance().Register("Gemm", BuildGemmOp);
    return true;
}();

}  // namespace

void EnsureGemmOperatorRegistered() { (void)g_gemm_op_registered; }

GemmOp::GemmOp() : OpBase("gemm", "Gemm") {}

GemmOp::GemmOp(const GemmParam& param) : GemmOp("gemm", param) {}

GemmOp::GemmOp(std::string name, const GemmParam& param) : OpBase(std::move(name), "Gemm"), param_(param) {
    SyncIO();
}

void GemmOp::SyncIO() {
    SetInputs({param_.a, param_.b, param_.bias});
    SetOutputs({param_.out});
}

bool GemmOp::ShapeCacheMatches() const {
    const bool bias_initialized = param_.bias != nullptr && param_.bias->IsInitialized();
    if (!shape_cache_valid_ || CheckShape() != 0 || param_.a->dims().data() != cached_a_dims_ ||
        param_.b->dims().data() != cached_b_dims_ ||
        (bias_initialized != cached_bias_initialized_) ||
        (bias_initialized && param_.bias->dims().data() != cached_bias_dims_) ||
        param_.out->dims().data() != cached_output_dims_ || !param_.out->IsInitialized()) {
        return false;
    }
    const auto dtype = ResolveExecutionDataType({param_.a, param_.b, param_.bias, param_.out}, DataType::FP32);
    const auto required_bytes = static_cast<size_t>(param_.out->numel()) * DataTypeBytes(dtype);
    return param_.out->memory_size() >= required_bytes;
}

void GemmOp::UpdateShapeCache() {
    cached_a_dims_ = param_.a->dims().data();
    cached_b_dims_ = param_.b->dims().data();
    cached_bias_initialized_ = param_.bias != nullptr && param_.bias->IsInitialized();
    cached_bias_dims_ = cached_bias_initialized_ ? param_.bias->dims().data() : std::vector<int64_t>{};
    cached_output_dims_ = param_.out->dims().data();
    shape_cache_valid_ = true;
}

int32_t GemmOp::CheckShape() const {
    if (param_.a == nullptr || param_.b == nullptr || param_.out == nullptr) {
        return -1;
    }
    if (param_.a->dims().size() < 2 || param_.b->dims().size() != 2 || param_.trans_a) {
        return -1;
    }
    const int64_t a_k = param_.a->dims()[param_.a->dims().size() - 1];
    const int64_t b_k = param_.trans_b ? param_.b->dims()[1] : param_.b->dims()[0];
    const int64_t out_n = param_.trans_b ? param_.b->dims()[0] : param_.b->dims()[1];
    if (a_k != b_k) {
        return -1;
    }
    if (param_.bias != nullptr && param_.bias->IsInitialized()) {
        if (IsVectorBias(param_.bias.get(), out_n)) {
            // A vector bias may carry singleton leading dimensions so it can
            // be broadcast over a batched decode activation.
        } else if (param_.bias->dims().size() == 2) {
            if (param_.a->dims().size() != 2) {
                return -1;
            }
            const int64_t out_m = param_.a->dims()[0];
            if (param_.bias->dims()[0] != out_m || param_.bias->dims()[1] != out_n) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

int32_t GemmOp::InferOutputShapes() {
    ++shape_inference_count_;
    shape_cache_valid_ = false;
    if (CheckShape() != 0) {
        return -1;
    }
    std::vector<int64_t> out_shape = param_.a->dims().data();
    out_shape.back() = param_.trans_b ? param_.b->dims()[0] : param_.b->dims()[1];
    const size_t required_bytes =
        static_cast<size_t>(std::accumulate(out_shape.begin(), out_shape.end(), int64_t{1}, std::multiplies<int64_t>())) *
        DataTypeBytes(ResolveExecutionDataType({param_.a, param_.b, param_.bias, param_.out}, DataType::FP32));
    if (param_.out == nullptr || !param_.out->IsInitialized() || param_.out->memory_size() < required_bytes) {
        param_.out = std::make_shared<Tensor>(out_shape);
    } else {
        param_.out->Resize(out_shape);
    }
    SyncIO();
    UpdateShapeCache();
    return 0;
}

int32_t GemmOp::Run() {
    if (!ShapeCacheMatches() && InferOutputShapes() != 0) {
        return -1;
    }
    if (kernel_ == nullptr) {
        return -1;
    }
    return kernel_->compute();
}

}  // namespace operators
}  // namespace feather
