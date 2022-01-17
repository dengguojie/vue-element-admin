/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021.All rights reserved.
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

#include "geometric.h"
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kGeometric = "Geometric";

#define Geometric_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                           \
    uint32_t result = DoCompute<TYPE>(CTX);                 \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("Geometric kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }
}  // namespace

namespace aicpu {
template <typename T>
uint32_t GeometricCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input_tensor = ctx.Input(0);
  auto input_tensor_shape = input_tensor->GetTensorShape();
  auto output_tensor = ctx.Output(0);
  auto output_y = reinterpret_cast<T *>(output_tensor->GetData());
  AttrValue *p = ctx.GetAttr("p");
  p_ = (p == nullptr) ? 0.5 : (p->GetFloat());
  AttrValue *attr_seed = ctx.GetAttr("attr_seed");
  attr_seed_ = (attr_seed == nullptr) ? 0 : (attr_seed->GetInt());
  srand(attr_seed_);
  KERNEL_LOG_DEBUG("%s Attr[p] value[%d]", kGeometric, p_);
  std::vector<int64_t> x_dim_size;
  for (int j = 0; j < input_tensor_shape->GetDims(); j++) {
    x_dim_size.push_back(input_tensor_shape->GetDimSize(j));
  }
  int64_t x_data_size = 1;
  for (int j = 0; j < int(x_dim_size.size()); j++) {
    x_data_size *= x_dim_size[j];
  }
  for (int64_t j = 0; j < x_data_size; j++) {
    double pV = (double)rand() / (double)RAND_MAX;
    long rnd = (int)(log(1 - pV) / log(1 - p_)) + 1;
    *(output_y + j) = (T)(rnd);
  }
  return KERNEL_STATUS_OK;
}

uint32_t GeometricCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"p"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Geometric Check Greater params failed.");
  KERNEL_HANDLE_ERROR(ExtraParamCheck(ctx),
                      "Geometric check params failed.")
  DataType input_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x] data type is [%s].", kGeometric,
                   DTypeStr(input_data_type).c_str());
  switch (input_data_type) {
    Geometric_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    Geometric_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      KERNEL_LOG_ERROR("Geometric kernel data type [%s] not support.",
                       DTypeStr(input_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t GeometricCpuKernel::ExtraParamCheck(CpuKernelContext &ctx) {
  DataType x_type = ctx.Input(0)->GetDataType();
  KERNEL_CHECK_FALSE((x_type == DT_FLOAT || x_type == DT_FLOAT16),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data type of [x] need be DT_FLOAT or DT_FLOAT16.")
  AttrValue *p = ctx.GetAttr("p");
  p_ = (p == nullptr) ? 0.5 : (p->GetFloat());
  KERNEL_CHECK_FALSE(
      (p_ < 1 && p_ > 0), KERNEL_STATUS_PARAM_INVALID,
      "The value of p must be in the range of(0, 1), but got: [%f]", p_);
  auto input0_datasize = ctx.Input(0)->GetDataSize();
  auto output_datasize = ctx.Output(0)->GetDataSize();
  KERNEL_CHECK_FALSE((input0_datasize == output_datasize),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data size of input0 [%d] need be same with "
                     "output0 [%d].",
                     input0_datasize, output_datasize)
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kGeometric, GeometricCpuKernel);
}  // namespace aicpu
