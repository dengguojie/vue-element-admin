/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All right reserved.
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
#include "non_zero_with_value_shape.h"

#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"

namespace {
const char *kNonZeroWithValueShape = "NonZeroWithValueShape";
const uint32_t INPUTS_NUM = 3;
const uint32_t OUTPUTS_NUM = 2;
}  // namespace
namespace aicpu {
uint32_t NonZeroWithValueShapeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, INPUTS_NUM, OUTPUTS_NUM),
                      "Check input and output number failed.");
  Tensor *count = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(count, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_data[2] failed.", kNonZeroWithValueShape);

  Tensor *out_value = ctx.Output(0);
  auto out_value_shape = out_value->GetTensorShape();
  int32_t count_num = static_cast<int32_t *>(count->GetData())[0];
  std::vector<int64_t> out_value_shape_values = {count_num};
  out_value_shape->SetDimSizes(out_value_shape_values);

  Tensor *out_index = ctx.Output(1);
  auto out_index_shape = out_index->GetTensorShape();
  std::vector<int64_t> out_index_shape_values = {2, count_num};
  out_index_shape->SetDimSizes(out_index_shape_values);

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNonZeroWithValueShape, NonZeroWithValueShapeCpuKernel);
}  // namespace aicpu
