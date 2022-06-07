/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef MATRIX_DIAG_PART_V3_KERNELS_H
#define MATRIX_DIAG_PART_V3_KERNELS_H

#include "cpu_kernel.h"        // CpuKernel基类以及注册宏定义

namespace aicpu {
 class MatrixDiagPartV3CpuKernel : public CpuKernel {
  public:
  MatrixDiagPartV3CpuKernel() = default;
  ~MatrixDiagPartV3CpuKernel() = default;
  virtual uint32_t Compute(CpuKernelContext &ctx) override;
  private:
  /**
  * @brief Init params
  * @param ctx cpu kernel context
  * @return status if success
  */
   uint32_t CheckParam(CpuKernelContext &ctx);
  /**
  * @brief Init params
  * @param ctx cpu kernel context
  * @return status if success
  */
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t MultiProcessFunc(CpuKernelContext &ctx,
                            int64_t upper_diag_index,
                            int64_t num_diags,
                            int64_t max_diag_len,
                            int64_t num_rows,
                            int64_t num_cols,
                            int64_t num_array);

  template <typename T>
  uint32_t SingleProcessFunc(CpuKernelContext &ctx,
                            int64_t upper_diag_index,
                            int64_t num_diags,
                            int64_t max_diag_len,
                            int64_t num_rows,
                            int64_t num_cols,
                            int64_t num_array);
  bool left_align_superdiagonal = true;
  bool left_align_subdiagonal = true;
};
} // namespace aicpu
#endif

