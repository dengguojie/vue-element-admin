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

#include "calc_buckets_limit_and_offset.h"

#include <algorithm>
#include <vector>

#include "utils/kernel_util.h"

namespace aicpu {

uint32_t CalcBucketsLimitAndOffsetCpuKernel::InitParams(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE(
      (ctx.GetInputsSize() == kInputNum), KERNEL_STATUS_PARAM_INVALID,
      "%s op need has %u inputs, but got %u inputs", kCalcBucketsLimitAndOffset,
      kInputNum, ctx.GetInputsSize());
  KERNEL_CHECK_FALSE(
      (ctx.GetOutputsSize() == kOutputNum), KERNEL_STATUS_PARAM_INVALID,
      "%s op need has %u outputs, but got %u outputs",
      kCalcBucketsLimitAndOffset, kOutputNum, ctx.GetOutputsSize());
  for (uint32_t i = 0; i < kInputNum; ++i) {
    auto input = ctx.Input(i);
    int64_t num_elements = input->NumElements();
    KERNEL_CHECK_FALSE(
        (num_elements >= 0), KERNEL_STATUS_PARAM_INVALID,
        "%s op input[%u] elements num should >= 0, but got [%lld]",
        kCalcBucketsLimitAndOffset, i, num_elements);
    int32_t *input_data = reinterpret_cast<int32_t *>(input->GetData());
    KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID,
                         "%s op input[%u] data is nullptr.",
                         kCalcBucketsLimitAndOffset, i);
    input_num_elements_[i] = num_elements;
    datas_[i] = input_data;
  }
  for (uint32_t i = 0; i < kOutputNum; ++i) {
    auto output = ctx.Output(i);
    int32_t *output_data = reinterpret_cast<int32_t *>(output->GetData());
    KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                         "%s op output[%u] data is nullptr.",
                         kCalcBucketsLimitAndOffset, i);
    datas_[kInputNum + i] = output_data;
  }
  auto attr = ctx.GetAttr("total_limit");
  KERNEL_CHECK_NULLPTR(attr, KERNEL_STATUS_PARAM_INVALID,
                       "%s op get total_limit attr failed.",
                       kCalcBucketsLimitAndOffset);
  total_limit_ = attr->GetInt();
  return KERNEL_STATUS_OK;
}

uint32_t CalcBucketsLimitAndOffsetCpuKernel::Compute(CpuKernelContext &ctx) {
  auto ret = InitParams(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  int32_t *counts = new int32_t[input_num_elements_[0]];
  for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
    if ((datas_[0][i] >= input_num_elements_[1]) ||
        (datas_[0][i] >= input_num_elements_[2])) {
      KERNEL_LOG_ERROR(
          "%s op input0[%lld] = %d is out of range input1 num elements [0, "
          "%lld) or input2 num elements [0, %lld).",
          kCalcBucketsLimitAndOffset, i, datas_[0][i], input_num_elements_[1],
          input_num_elements_[2]);
      delete[] counts;
      return KERNEL_STATUS_PARAM_INVALID;
    }
    counts[i] = datas_[1][datas_[0][i]];
    datas_[3][i] = counts[i];
    datas_[4][i] = datas_[2][datas_[0][i]];
  }
  std::sort(counts, counts + input_num_elements_[0]);
  int64_t rest = total_limit_;
  int64_t limit = 0;
  for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
    limit = rest / (input_num_elements_[0] - i);
    if (counts[i] > limit) {
      break;
    }
    rest -= counts[i];
  }
  for (int64_t i = 0; i < input_num_elements_[0]; ++i) {
    if (static_cast<int64_t>(datas_[3][i]) > limit) {
      datas_[3][i] = static_cast<int32_t>(limit);
    }
  }
  delete[] counts;
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCalcBucketsLimitAndOffset,
                    CalcBucketsLimitAndOffsetCpuKernel);
}  // namespace aicpu