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

#ifndef _AICPU_GET_DYNAMIC_DIMS_KERNELS_H_
#define _AICPU_GET_DYNAMIC_DIMS_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class GetDynamicDimsCpuKernel : public CpuKernel {
public:
  ~GetDynamicDimsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

private:
  // get each input's configured shape from attr shape_info
  std::vector<std::vector<int64_t>>
      GetShapeInfos(std::vector<int64_t> &shape_info) const;

  // get each input's practical shape from inputs
  uint32_t GetInputShapes(CpuKernelContext &ctx,
      std::vector<std::vector<int64_t>> &input_shapes) const;
};
} // namespace aicpu

#endif
