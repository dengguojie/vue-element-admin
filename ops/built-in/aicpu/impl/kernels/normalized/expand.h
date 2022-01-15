/***
 *  Copyright Huawei Technologies Co., Ltd. 2021-2021.All rights reserved.
 *  Description:This file provides the function of expandding.
 *  Author: Huawei.
 *  Create:2021-10-08.
 ***/
#ifndef AICPU_KERNELS_NORMALIZED_EXPAND_H
#define AICPU_KERNELS_NORMALIZED_EXPAND_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "Eigen/Core"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace aicpu {
class ExpandCpuKernel : public CpuKernel {
 public:
  ExpandCpuKernel() = default;
  ~ExpandCpuKernel() = default;
  uint32_t Compute(CpuKernelContext& ctx) override;

 private:
};
}  // namespace aicpu

#endif