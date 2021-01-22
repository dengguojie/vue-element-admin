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
#ifndef AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_V2_H_
#define AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_V2_H_

#include "cpu_kernel.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class StridedSliceV2CpuKernel : public CpuKernel {
 public:
  ~StridedSliceV2CpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckParam(const Tensor *begin, const Tensor *end,
                      const Tensor *axes, const Tensor *strides);

  template <typename T>
  uint32_t BuildBeginParam(const std::shared_ptr<TensorShape> &x_shape,
                           const Tensor *begin,
                           std::vector<int64_t> &begin_vec);

  template <typename T>
  uint32_t BuildEndParam(const std::shared_ptr<TensorShape> &x_shape,
                         const Tensor *end, std::vector<int64_t> &end_vec);

  template <typename T>
  uint32_t BuildStridesParam(const std::shared_ptr<TensorShape> &x_shape,
                             const Tensor *strides,
                             std::vector<int64_t> &strides_vec);

  template <typename T>
  uint32_t BuildAxesParam(const std::shared_ptr<TensorShape> &x_shape,
                          const Tensor *axes, std::vector<int64_t> &axes_vec);

  template <typename T>
  uint32_t BuildParam(const Tensor *x, const Tensor *begin, const Tensor *end,
                      const Tensor *axes, const Tensor *strides,
                      std::vector<int64_t> &begin_vec,
                      std::vector<int64_t> &end_vec,
                      std::vector<int64_t> &strides_vec);

  template <typename T>
  uint32_t CheckAndBuildParam(const Tensor *x, const Tensor *begin,
                              const Tensor *end, const Tensor *axes,
                              const Tensor *strides,
                              std::vector<int64_t> &begin_vec,
                              std::vector<int64_t> &end_vec,
                              std::vector<int64_t> &strides_vec);

  uint32_t DoStridedSliceV2(CpuKernelContext &ctx,
                            const std::vector<int64_t> &begin_vec,
                            const std::vector<int64_t> &end_vec,
                            const std::vector<int64_t> &strides_vec);
};
}  // namespace aicpu
#endif
