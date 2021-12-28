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

#include "exponential.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cmath"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kExponential = "Exponential";

#define EXPONENTIAL_COMPUTE_CASE(DTYPE, TYPE, CTX)              \
  case (DTYPE): {                                               \
    uint32_t result = ExponentialCompute<TYPE>(CTX);            \
    if (result != KERNEL_STATUS_OK) {                           \
      KERNEL_LOG_ERROR("Exponential kernel compute failed.");   \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}

namespace aicpu {
uint32_t ExponentialCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Exponential check input and output number failed.");

  Tensor *input = ctx.Input(0);
  auto data_type = input->GetDataType();

  Tensor *output = ctx.Output(0);

  AttrValue *attr_seed = ctx.GetAttr("seed");
  seed_ = (attr_seed == nullptr) ? 0 : (attr_seed->GetInt());

  AttrValue *attr_lambda = ctx.GetAttr("lambda");
  lambda_ = (attr_lambda == nullptr) ? 1.0 : (attr_lambda->GetFloat());
  KERNEL_CHECK_FALSE((lambda_ > 0),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The value of lambda must be greater than 0, but got: [%f]", lambda_);

  if (data_type != output->GetDataType()) {
    KERNEL_LOG_ERROR(
        "Exponential kernel data type need be same, input_data_dtype is [%s], "
        "output_data_type is [%s]. Support data_types: DT_FLOAT16, DT_FLOAT, DT_DOUBLE.",
        DTypeStr(data_type).c_str(), DTypeStr(output->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // choose random data generate function depend on dataType
  switch (data_type) {
    EXPONENTIAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    EXPONENTIAL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    EXPONENTIAL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Exponential kernel data type [%s] not support. "
                       "Support data_types: DT_FLOAT16, DT_FLOAT, DT_DOUBLE.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ExponentialCpuKernel::ExponentialCompute(CpuKernelContext &ctx) {
  auto input_x = ctx.Input(0);
  auto output_tensor = ctx.Output(0);

  Eigen::Tensor<T, 1> eigen_input_x(input_x->GetTensorShape()->NumElements());
  eigen_input_x.setConstant(static_cast<T>(1.0));

  Eigen::Tensor<T, 1> eigen_input_lambda(input_x->GetTensorShape()->NumElements());
  eigen_input_lambda.setConstant(static_cast<T>(lambda_));

  Eigen::TensorMap<Eigen::Tensor<T, 1>> eigen_output(
      static_cast<T *>(output_tensor->GetData()),
      output_tensor->GetTensorShape()->NumElements());

  Eigen::Tensor<T, 1> eigen_random(output_tensor->GetTensorShape()->NumElements());

  eigen_random = eigen_output.random(Eigen::internal::UniformRandomGenerator<T>(seed_));

  eigen_output =  - (eigen_input_x - eigen_random).log() / eigen_input_lambda;

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kExponential, ExponentialCpuKernel);
}  // namespace aicpu