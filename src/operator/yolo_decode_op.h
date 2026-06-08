#ifndef FEATHER_OPERATOR_YOLO_DECODE_OP_H
#define FEATHER_OPERATOR_YOLO_DECODE_OP_H

#include <memory>
#include <string>

#include "core/operator.h"
#include "src/kernel/yolo_decode.h"
#include "src/operator/params.h"

namespace feather {
namespace operators {

class YoloDecodeOp : public OpBase {
   public:
    YoloDecodeOp();
    explicit YoloDecodeOp(const YoloDecodeParam& param);
    YoloDecodeOp(std::string name, const YoloDecodeParam& param);

    int32_t CheckShape() const override;
    int32_t InferOutputShapes() override;
    void AttachKernel(std::unique_ptr<KernelBase> kernel) override {
        kernel_ = std::move(kernel);
        if (kernel_ != nullptr) {
            kernel_->SetParam((void*)&param_);
        }
    }
    std::unique_ptr<KernelBase> DetachKernel() override { return std::move(kernel_); }
    bool HasKernel() const override { return kernel_ != nullptr; }
    int32_t Run() override;

   private:
    void SyncIO();

    YoloDecodeParam param_;
    std::unique_ptr<KernelBase> kernel_;
};

}  // namespace operators
}  // namespace feather

#endif  // FEATHER_OPERATOR_YOLO_DECODE_OP_H
