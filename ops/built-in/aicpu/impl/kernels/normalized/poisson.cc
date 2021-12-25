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

#include "poisson.h"

#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kPoisson = "Poisson";

#define Poisson_COMPUTE_CASE(DTYPE, TYPE, CTX)                \
  case (DTYPE): {                                             \
    uint32_t result = PoissonCompute<TYPE>(CTX);              \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("Poisson kernel compute failed.");     \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }		
}

namespace aicpu {
template <typename T>
uint32_t PoissonCpuKernel::PoissonCompute(CpuKernelContext &ctx) {
  auto input_x = ctx.Input(0);
  auto x= reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t final_seed=0;
  auto attr_seed = ctx.GetAttr("seed");
  if (attr_seed != nullptr) {
    final_seed = attr_seed->GetInt();
  }
  srand(final_seed);

  for(uint32_t j = 0; j < input_x->NumElements(); j++){
    long k=0;
    double p=1.0;
    T r=*(x+j);
    double l=exp(-r);
    
    while(p>=l)
    {
      double u = (double)rand() / RAND_MAX;
      p *= u;
      k++;
    }
      *(y + j) = (T)(k-1);
  }
  return KERNEL_STATUS_OK;
}

uint32_t PoissonCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
		                  "Poisson Check input and output failed.");
  KERNEL_HANDLE_ERROR(PoissonParamCheck(ctx), "Poissonon check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    Poisson_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    Poisson_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      KERNEL_LOG_ERROR("Poisson kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t PoissonCpuKernel::PoissonParamCheck(CpuKernelContext &ctx) {
  DataType x_type = ctx.Input(0)->GetDataType();
  KERNEL_CHECK_FALSE((x_type == DT_FLOAT || x_type == DT_FLOAT16),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data type of [x] need be DT_FLOAT or DT_FLOAT16.")
  auto input0_datasize = ctx.Input(0)->GetDataSize();
  auto output_datasize = ctx.Output(0)->GetDataSize();
  KERNEL_CHECK_FALSE((input0_datasize == output_datasize),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data size of input0 [%d] need be same with "
                     "output0 [%d].",
                     input0_datasize, output_datasize);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kPoisson, PoissonCpuKernel);
}  //namespace aicpu