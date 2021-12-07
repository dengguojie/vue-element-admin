/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 */

#ifndef AICPU_KERNELS_DEVICE_TILE_WITH_AXIS_H_
#define AICPU_KERNELS_DEVICE_TILE_WITH_AXIS_H_

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class TileWithAxisCpuKernel : public CpuKernel {
 public:
  TileWithAxisCpuKernel() = default;
  ~TileWithAxisCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t TileParaCheck(CpuKernelContext &ctx);

  template <typename T, int32_t OPTION, int32_t DIMS>
  uint32_t TileComputeByAxis(CpuKernelContext &ctx);

  template <typename T, int32_t OPTION>
  uint32_t TileComputeInDims(CpuKernelContext &ctx);

  template <typename T>
  uint32_t TileCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEVICE_TILE_WITH_AXIS_H_