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
#include "random_gamma.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kRandomGamma = "RandomGamma";

#define RANDOM_Gamma_GENERATE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                               \
    Generate<TYPE>(ctx, output);                \
    break;                                      \
  }

#define RANDOM_Gamma_EIGEN_TENSOR_ASSIGN_CASE(ALIGNMENT_TYPE)         \
  Eigen::TensorMap<Eigen::Tensor<T, 1>, ALIGNMENT_TYPE> eigen_output( \
      static_cast<T *>(output->GetData()),                            \
      output->GetTensorShape()->NumElements());                       \
  GammaRandomGenerator<T> m_generator(alpha_flat[0], final_seed);     \
  for (int i = 0; i < num_of_alpha; i++) {                            \
    m_generator.setAlpha(alpha_flat[i]);                              \
    for (int j = i; j < num_of_output; j += num_of_alpha) {           \
      eigen_output(j) = m_generator.gen();                            \
    }                                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t RandomGammaCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "RandomGamma check input and output number failed.");
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  auto data_type = input_1->GetDataType();
  Tensor *output = ctx.Output(kFirstOutputIndex);
  output->SetDataType(data_type);

  // choose random data generate function depend on dataType
  switch (data_type) {
    RANDOM_Gamma_GENERATE_CASE(DT_FLOAT16, Eigen::half)
        RANDOM_Gamma_GENERATE_CASE(DT_FLOAT, float)
            RANDOM_Gamma_GENERATE_CASE(DT_DOUBLE, double) default
        : KERNEL_LOG_ERROR("RandomGamma kernel data type [%s] not support.",
                           DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void RandomGammaCpuKernel::Generate(CpuKernelContext &ctx, Tensor *output) {
  uint64_t final_seed = 0;
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

  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  double *alpha_flat = static_cast<double *>(input_1->GetData());
  int64_t num_of_alpha = input_1->NumElements();
  int64_t num_of_output = output->GetTensorShape()->NumElements();

  if (AddrAlignedCheck(output->GetData())) {
    RANDOM_Gamma_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Aligned);
  } else {
    RANDOM_Gamma_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Unaligned);
  }
}

REGISTER_CPU_KERNEL(kRandomGamma, RandomGammaCpuKernel);
}  // namespace aicpu
