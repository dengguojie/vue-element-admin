/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _AICPU_TEST_ADD_KERNELS_H_
#define _AICPU_TEST_ADD_KERNELS_H_

#include "Eigen/Core"
#include "cpu_kernel.h"

namespace aicpu {
struct tensorShapeDesc {
  int64_t batch = 0;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct extraParameters {
  int32_t strideH = 0;
  int32_t strideW = 0;
  int32_t padUp = 0;
  int32_t padDown = 0;
  int32_t padLeft = 0;
  int32_t padRight = 0;
  int32_t ksizeX = 0;
  int32_t ksizeY = 0;
  int32_t dilationsH = 0;
  int32_t dilationsW = 0;
};

class DeformableOffsetsCpuKernel : public CpuKernel {
 public:
  DeformableOffsetsCpuKernel() = default;
  ~DeformableOffsetsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputParam(CpuKernelContext &ctx);
  uint32_t ParseInputParam();
  uint32_t CheckInputParam();

  template <typename T>
  uint32_t DoCompute(Eigen::half *inputDataX, T *inputDataOffsets,
                     Eigen::half *inputDataY);

  template <typename T>
  uint32_t ComputePosition(const Eigen::half *inputX, const T *inputOffset,
                           Eigen::half *inputY, int currentAxis);

  template <typename T>
  uint32_t ComputeResult(const Eigen::half *inputX, const T *inputOffset,
                         Eigen::half *inputY, int xSrc, int ySrc,
                         int currentAxis);

  uint32_t BilinearInterpolate(Eigen::half &out, const Eigen::half *in,
                               int c_axis, float h, float w);

 private:
  Tensor *x_tensor = nullptr;
  Tensor *offsets_tensor = nullptr;
  Tensor *y_tensor = nullptr;
  // Move step size for convolution calculation
  std::vector<int64_t> stride_list_;
  // Specify the number of layers filled with 0 around the input x feature map
  std::vector<int64_t> pads_list_;
  // Convolution kernel size
  std::vector<int64_t> ksize_list_;
  // Used to change the size of the convolution kernel
  std::vector<int64_t> dilation_list_;
  // Specify the type of input x
  std::string data_format_;
  int deformable_groups_ = 1;
  tensorShapeDesc x_;
  tensorShapeDesc offset_;
  tensorShapeDesc y_;
  extraParameters param_;
};
}
#endif
