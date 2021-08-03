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
#include "neg.h"

#include <complex>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char *kNeg = "Neg";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t NegCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto output_tensor = ctx.Output(0);
  DataType input_type = input0_tensor->GetDataType();
  DataType output_type = output_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_INNER_ERROR,
                     "Input data type[%s], output data type[%s] "
                     "must be same",
                     DTypeStr(input_type).c_str(),
                     DTypeStr(output_type).c_str());
  auto input0_elements_num = input0_tensor->NumElements();
  TensorMap<T> input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                      input0_elements_num);
  auto output_elements_num = output_tensor->NumElements();
  TensorMap<T> output(reinterpret_cast<T *>(output_tensor->GetData()),
                      output_elements_num);
  output = -input0;
  return KERNEL_STATUS_OK;
}

uint32_t NegCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Neg params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x] data type is [%s].", kNeg,
                   DTypeStr(input0_data_type).c_str());
  switch (input0_data_type) {
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<complex<double>>(ctx);
    default:
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kNeg, NegCpuKernel);

}  // namespace aicpu