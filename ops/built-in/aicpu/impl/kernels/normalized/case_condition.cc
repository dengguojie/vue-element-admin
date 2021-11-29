/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "case_condition.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kCaseCondition = "CaseCondition";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;

template <typename T>
void CaseCondition(const T i, const T j, const T k, int32_t *output_data) {
  if (i >= k && j < k) {
    *output_data = 0;
  } else if (i == k && j == k) {
    *output_data = 1;
  } else if (i > k && j == k) {
    *output_data = 2;
  } else if (i == k && j > k) {
    *output_data = 3;
  } else if (i > k && j > k) {
    *output_data = 4;
  } else {
    *output_data = 5;
  }
  KERNEL_LOG_INFO("%s param value, i=[%ld], j=[%ld], k=[%ld], output=[%d]",
                  kCaseCondition, static_cast<int64_t>(i),
                  static_cast<int64_t>(j), static_cast<int64_t>(k),
                  *output_data);
}
}  // namespace

namespace aicpu {
uint32_t CaseConditionCpuKernel::Check(const Tensor *x, const Tensor *output) {
  DataType x_type = x->GetDataType();
  switch (x_type) {
    case DT_INT32:
    case DT_INT64:
    case DT_UINT64:
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input x data type[%s]",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  // x[0] is i, x[1] is j, x[2] is k
  KERNEL_CHECK_FALSE((IsVector(x->GetTensorShape()->GetDimSizes()) &&
                      x->NumElements() == 3), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x] must be a vector with shape=[3]")

  DataType output_type = output->GetDataType();
  KERNEL_CHECK_FALSE((output_type == DT_INT32), KERNEL_STATUS_INNER_ERROR,
                     "Output[y] data type[%s] must be DT_INT32",
                     DTypeStr(output_type).c_str())
  return KERNEL_STATUS_OK;
}

uint32_t CaseConditionCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"algorithm"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Check CaseCondition params failed.");
  KERNEL_CHECK_FALSE((ctx.GetAttr("algorithm")->GetString() == "LU"),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Attr[algorithm] only support LU")
  Tensor *x = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  auto x_data = x->GetData();
  auto output_data = static_cast<int32_t *>(output->GetData());

  uint32_t ret = Check(x, output);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
                     "Check param failed, ret=[%u].", ret);

  switch (x->GetDataType()) {
    case DT_INT32:
      CaseCondition(static_cast<int32_t *>(x_data)[0],
                    static_cast<int32_t *>(x_data)[1],
                    static_cast<int32_t *>(x_data)[2], output_data);
      break;
    case DT_INT64:
      CaseCondition(static_cast<int64_t *>(x_data)[0],
                    static_cast<int64_t *>(x_data)[1],
                    static_cast<int64_t *>(x_data)[2], output_data);
      break;
    case DT_UINT64:
      CaseCondition(static_cast<uint64_t *>(x_data)[0],
                    static_cast<uint64_t *>(x_data)[1],
                    static_cast<uint64_t *>(x_data)[2], output_data);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input x data type[%s]",
                       DTypeStr(x->GetDataType()).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kCaseCondition, CaseConditionCpuKernel);
}  // namespace aicpu
