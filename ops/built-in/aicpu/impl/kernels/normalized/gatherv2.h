#ifndef AICPU_KERNELS_NORMALIZED_GATHERV2_H
#define AICPU_KERNELS_NORMALIZED_GATHERV2_H

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
class GatherV2CpuKernel : public CpuKernel {
  public:
    GatherV2CpuKernel() = default;
    ~GatherV2CpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;
  private:
    uint32_t GetInputAndCheck(CpuKernelContext &ctx);
};
} // namespace aicpu
#endif