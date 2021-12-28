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

#include "combinations.h"

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t ATTRRDEFAULT = 2;
const char *kCombinations = "Combinations";

#define COMBINATIONS_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                              \
    uint32_t result = DoCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("Combinations kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t CombinationsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Greater params failed.");
  KERNEL_HANDLE_ERROR(CombinationsParamCheck(ctx),
                      "Combinations check params failed.");
  DataType input_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x] data type is [%s].", kCombinations,
                   DTypeStr(input_data_type).c_str());
  switch (input_data_type) {
    COMBINATIONS_COMPUTE_CASE(DT_BOOL, bool, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_FLOAT, float, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    COMBINATIONS_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    default:
      KERNEL_LOG_ERROR("Combinations kernel data type [%s] not support.",
                       DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CombinationsCpuKernel::CombinationsParamCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  DataType input_type = input->GetDataType();
  DataType output_type = output->GetDataType();
  KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output [%s] need be same with "
                     "input [%s].",
                     DTypeStr(output_type).c_str(),
                     DTypeStr(input_type).c_str())
  KERNEL_LOG_DEBUG(
      "CombinationsCpuKernel[%s], input: size[%llu]; output: size[%llu].",
      ctx.GetOpType().c_str(), input->GetDataSize(), output->GetDataSize());
  int32_t input_num = ctx.Input(0)->NumElements();
  AttrValue *r_ = ctx.GetAttr("r");
  auto r = (r_ == nullptr) ? ATTRRDEFAULT : (r_->GetInt());
  KERNEL_CHECK_FALSE(
      (r >= 1), KERNEL_STATUS_PARAM_INVALID,
      "The value of r must be greater than or equal to 1, but got: [%d]", r);
  KERNEL_CHECK_FALSE((r <= input_num), KERNEL_STATUS_PARAM_INVALID,
                     "The value of r must be smaller than or equal to "
                     "input_num, but got: [%d]",
                     r);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CombinationsCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_tensor = ctx.Input(0);
  auto inputShape = input_tensor->GetTensorShape();
  KERNEL_LOG_DEBUG("%s input_tensor dims[%d]", kCombinations,
                   inputShape->GetDims());
  KERNEL_CHECK_FALSE(inputShape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input must be 1D.")
  AttrValue *r_ = ctx.GetAttr("r");
  auto r = (r_ == nullptr) ? ATTRRDEFAULT : (r_->GetInt());
  KERNEL_LOG_DEBUG("%s Attr[r] value[%d]", kCombinations, r);
  AttrValue *with_replacement_ = ctx.GetAttr("with_replacement");
  auto with_replacement =
      (with_replacement_ == nullptr) ? false : (with_replacement_->GetBool());
  KERNEL_LOG_DEBUG("%s Attr[with_replacement] value[%d]", kCombinations,
                   with_replacement);
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t input_num = ctx.Input(0)->NumElements();
  std::vector<int> t(r, 0);
  T *combination = new T[r];
  // The combination is used to temporarily store the addresses of the elements
  // in the input so that the combined address can be stored in the output after
  // the group has been arranged.
  int k = 0;
  int64_t idx = 0;
  while (idx >= 0) {
    *(combination + idx) = *(input + t[idx]);
    t[idx]++;
    if (t[idx] > input_num) {
      idx--;
    } else if (idx == r - 1) {
      for (int j = 0; j < r; j++) {
        *(output + k) = *(combination + j);
        k++;
      }
    } else {
      idx++;
      t[idx] = (with_replacement) ? (t[idx - 1] - 1) : t[idx - 1];
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCombinations, CombinationsCpuKernel);
}  // namespace aicpu