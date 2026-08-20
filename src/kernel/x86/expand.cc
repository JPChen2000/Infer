#include "src/kernel/expand.h"

#include <cstring>
#include <vector>

#include "src/kernel/common/tensor_op_utils.h"
#include "util/timer.h"

namespace feather {
namespace kernel {

namespace {

bool ReadAndValidateShape(feather::operators::ExpandParam* param, std::vector<int64_t>* target_shape) {
    if (param == nullptr || param->input == nullptr || param->shape == nullptr || param->out == nullptr ||
        param->input->data_type() != DataType::BF16 || !param->input->IsInitialized() ||
        !param->out->IsInitialized() ||
        (param->shape->data_type() != DataType::INT64 && param->shape->data_type() != DataType::INT32) ||
        target_shape == nullptr) {
        return false;
    }
    target_shape->clear();
    target_shape->reserve(static_cast<size_t>(param->shape->numel()));
    for (int64_t index = 0; index < param->shape->numel(); ++index) {
        target_shape->push_back(common_tensor_detail::ReadInteger(param->shape.get(), index));
    }
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    if (target_shape->size() < input_dims.size() || target_shape->size() != output_dims.size()) {
        return false;
    }
    const size_t rank_gap = target_shape->size() - input_dims.size();
    for (size_t axis = 0; axis < target_shape->size(); ++axis) {
        const int64_t input_dim = axis < rank_gap ? 1 : input_dims[axis - rank_gap];
        const int64_t target_dim = (*target_shape)[axis];
        if (input_dim <= 0 || target_dim <= 0 ||
            (input_dim != target_dim && input_dim != 1 && target_dim != 1) ||
            output_dims[axis] != std::max(input_dim, target_dim)) {
            return false;
        }
    }
    return param->out->memory_size() >= static_cast<size_t>(param->out->numel()) * sizeof(uint16_t);
}

int32_t ComputeExpandFallbackBf16(feather::operators::ExpandParam* param) {
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    const auto output_strides = common_tensor_detail::ComputeStrides(output_dims);
    const auto input_strides = common_tensor_detail::ComputeStrides(input_dims);
    std::vector<int64_t> coordinates;
    const auto* input = static_cast<const uint16_t*>(param->input->raw_data());
    auto* output = static_cast<uint16_t*>(param->out->raw_data());
    for (int64_t linear = 0; linear < param->out->numel(); ++linear) {
        common_tensor_detail::LinearToCoords(linear, output_dims, output_strides, &coordinates);
        output[linear] = input[common_tensor_detail::BroadcastOffset(coordinates, input_dims, input_strides)];
    }
    return 0;
}

bool FindSingleContiguousBroadcastAxis(const std::vector<int64_t>& input_dims,
                                       const std::vector<int64_t>& output_dims, int64_t* outer, int64_t* repeat,
                                       int64_t* inner) {
    if (outer == nullptr || repeat == nullptr || inner == nullptr || input_dims.size() != output_dims.size()) {
        return false;
    }
    int64_t axis = -1;
    for (size_t index = 0; index < input_dims.size(); ++index) {
        if (input_dims[index] == output_dims[index]) {
            continue;
        }
        if (input_dims[index] != 1 || output_dims[index] <= 1 || axis >= 0) {
            return false;
        }
        axis = static_cast<int64_t>(index);
    }
    if (axis < 0) {
        return false;
    }
    *outer = 1;
    *inner = 1;
    for (int64_t index = 0; index < axis; ++index) *outer *= input_dims[static_cast<size_t>(index)];
    for (size_t index = static_cast<size_t>(axis + 1); index < input_dims.size(); ++index) {
        *inner *= input_dims[index];
    }
    *repeat = output_dims[static_cast<size_t>(axis)];
    return true;
}

int32_t ComputeExpandBf16(feather::operators::ExpandParam* param) {
    std::vector<int64_t> target_shape;
    if (!ReadAndValidateShape(param, &target_shape)) {
        return -1;
    }
    const auto& input_dims = param->input->dims().data();
    const auto& output_dims = param->out->dims().data();
    param->out->set_data_type(DataType::BF16);
    const auto* input = static_cast<const uint16_t*>(param->input->raw_data());
    auto* output = static_cast<uint16_t*>(param->out->raw_data());

    int64_t outer = 0;
    int64_t repeat = 0;
    int64_t inner = 0;
    if (FindSingleContiguousBroadcastAxis(input_dims, output_dims, &outer, &repeat, &inner)) {
        const size_t block_bytes = static_cast<size_t>(inner) * sizeof(uint16_t);
        for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
            const auto* source = input + outer_index * inner;
            auto* destination = output + outer_index * repeat * inner;
            for (int64_t copy = 0; copy < repeat; ++copy) {
                std::memcpy(destination + copy * inner, source, block_bytes);
            }
        }
        return 0;
    }

    if (param->input->numel() == param->out->numel()) {
        std::memcpy(output, input, static_cast<size_t>(param->input->numel()) * sizeof(uint16_t));
        return 0;
    }
    return ComputeExpandFallbackBf16(param);
}

}  // namespace

template <>
int32_t ExpandKernel<DeviceType::X86, DataType::BF16>::compute() {
    AutoTimer timer("X86::Expand::BF16");
    return ComputeExpandBf16(static_cast<feather::operators::ExpandParam*>(param_));
}

void EnsureX86ExpandKernelsRegistered() {
    static bool registered = []() {
        KernelDispatcher::instance().registerKernel(
            DeviceType::X86, DataType::BF16, "Expand",
            []() { return std::make_unique<ExpandKernel<DeviceType::X86, DataType::BF16>>(); });
        return true;
    }();
    (void)registered;
}

}  // namespace kernel
}  // namespace feather
