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
#ifndef AICPU_KERNELS_NORMALIZED_SPATIAL_TRANSFORMER_H_
#define AICPU_KERNELS_NORMALIZED_SPATIAL_TRANSFORMER_H_

#include <memory>
#include <vector>

#include "securec.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class SpatialTransformerCpuKernel : public CpuKernel {
 public:
  ~SpatialTransformerCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
 private:
  /**
   * @brief Init params and check valid
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t GetInputAndCheckValid(CpuKernelContext &ctx);

  /**
   * @brief compute for all types
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T> uint32_t DoCompute(CpuKernelContext &ctx);

  /**
   * @brief compute for NCHW format
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T, typename T1> uint32_t DoCompute4D(CpuKernelContext &ctx);

  /**
   * @brief compute for NC1HWC0 format
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T, typename T1> uint32_t DoCompute5D(CpuKernelContext &ctx);

  /**
   * @brief compute for NC1HWC0 format
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T, typename T1> uint32_t DoCompute5D_C1(CpuKernelContext &ctx);

 private:
  Tensor* input_tensor_ = nullptr;
  Tensor* input_theta_ = nullptr;
  Tensor* output_tensor_ = nullptr;
  int32_t input_n_ = 0;
  int32_t input_c_ = 0;
  int32_t input_c1_ = 0;
  int32_t input_c0_ = 0;
  int32_t input_h_ = 0;
  int32_t input_w_ = 0;
  int32_t output_h_ = 0;
  int32_t output_w_ = 0;
  int32_t stn_ori_channel_ = 0;
  std::vector<float> theta_;
  std::vector<int64_t> theta_valid_;
  Format date_format_ = FORMAT_ND;
  DataType input_data_type_ = DT_FLOAT;
  DataType input_theta_type_ = DT_FLOAT;
  DataType output_data_type_ = DT_FLOAT;
};
}  // namespace aicpu
#endif