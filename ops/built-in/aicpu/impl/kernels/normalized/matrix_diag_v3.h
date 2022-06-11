/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved. 
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
#ifndef AICPU_KERNELS_NORMALIZED_MATRIX_DIAG_V3_H_
#define AICPU_KERNELS_NORMALIZED_MATRIX_DIAG_V3_H_

#include "cpu_kernel.h"

namespace aicpu {
class MatrixDiagV3CpuKernel : public CpuKernel {
public:
  ~MatrixDiagV3CpuKernel() = default;
  virtual uint32_t Compute(CpuKernelContext &ctx) override;

private:
  uint32_t CheckParam(CpuKernelContext &ctx);
  std::pair<int, int> ComputeDiagLenAndContentOffset(
  int diag_index, int max_diag_len, int num_rows, int num_cols,
  bool left_align_superdiagonal, bool left_align_subdiagonal);
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t AdjustRowsAndCols(int32_t &num_rows,
                            int32_t &num_cols,
                            int32_t min_num_rows,
                            int32_t min_num_cols);
  uint32_t GetDiagIndex(CpuKernelContext &ctx,
                       int32_t &lower_diag_index,
                       int32_t &upper_diag_index,
                       int32_t &num_rows,
                       int32_t &num_cols);
  bool left_align_superdiagonal = true;
  bool left_align_subdiagonal = true;
};
} // namespace aicpu
#endif
