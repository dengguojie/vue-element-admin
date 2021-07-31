/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H_
#define AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H_

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class NonMaxSuppressionV3CpuKernel : public CpuKernel {
 public:
  ~NonMaxSuppressionV3CpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  template <typename T>
  static inline T IOUSimilarity(const T *box_1, const T *box_2);
  template <typename T, typename T_threshold>
  uint32_t DoCompute();

  const Tensor *boxes_ = nullptr;
  Tensor *scores_ = nullptr;
  Tensor *iou_threshold_tensor_ = nullptr;
  Tensor *score_threshold_tensor_ = nullptr;
  Tensor *output_indices_ = nullptr;
  int32_t num_boxes_ = 0;
  int32_t max_output_size_ = 0;
  DataType threshold_dtype_ = DT_UINT32;
  DataType boxes_scores_dtype_ = DT_UINT32;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H_
