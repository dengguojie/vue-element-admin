#ifndef _AICPU_UNIQUE_CUST_KERNELS_H_
#define _AICPU_UNIQUE_CUST_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class UniqueCpuKernel : public CpuKernel {
 public:
  ~UniqueCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif
