/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All right reserved.
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
#include "random_uniform.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kRandomUniform = "RandomUniform";

#define RANDOM_UNIFORM_GENERATE_CASE(DTYPE, TYPE)                     \
  case (DTYPE): {                                                     \
    Generate<TYPE>(ctx, output);                                      \
    break;                                                            \
  }

#define RANDOM_UNIFORM_EIGEN_TENSOR_ASSIGN_CASE(ALIGNMENT_TYPE)       \
  Eigen::TensorMap<Eigen::Tensor<T, 1>, ALIGNMENT_TYPE> eigen_output( \
      static_cast<T *>(output->GetData()),                            \
      output->GetTensorShape()->NumElements());                       \
  eigen_output.device(device) = eigen_output.random(                  \
      Eigen::internal::UniformRandomGenerator<T>(final_seed));
}

namespace aicpu {
uint32_t RandomUniformCpuKernel::Compute(CpuKernelContext &ctx) {
  auto attr_value = ctx.GetAttr("dtype");
  KERNEL_CHECK_NULLPTR(attr_value, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr[dtype] failed")
  auto data_type = static_cast<DataType>(attr_value->GetDataType());

  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  if (data_type != output->GetDataType()) {
    KERNEL_LOG_ERROR(
        "RandomUniform kernel data type not matched, dtype is [%u], "
        "out_data_type is [%u].",
        data_type, output->GetDataType());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose random data generate function depend on dataType
  switch (data_type) {
    RANDOM_UNIFORM_GENERATE_CASE(DT_FLOAT16, Eigen::half)
    RANDOM_UNIFORM_GENERATE_CASE(DT_FLOAT, float)
    RANDOM_UNIFORM_GENERATE_CASE(DT_DOUBLE, double)
    default:
      KERNEL_LOG_ERROR("RandomUniform kernel data type [%u] not support.",
                       data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void RandomUniformCpuKernel::Generate(CpuKernelContext &ctx, Tensor *output) {
  int64_t final_seed = 0;
  auto attr_seed = ctx.GetAttr("seed");
  if (attr_seed != nullptr) {
    final_seed = attr_seed->GetInt();
  }
  if (final_seed == 0) {
    auto attr_seed2 = ctx.GetAttr("seed2");
    if (attr_seed2 != nullptr) {
      final_seed = attr_seed2->GetInt();
    }
  }

  Eigen::ThreadPool pool(kThreadNum);
  Eigen::ThreadPoolDevice device(&pool, kThreadNum);
  if (AddrAlignedCheck(output->GetData())) {
    RANDOM_UNIFORM_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Aligned);
  } else {
    RANDOM_UNIFORM_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Unaligned);
  }
}

REGISTER_CPU_KERNEL(kRandomUniform, RandomUniformCpuKernel);
}  // namespace aicpu
