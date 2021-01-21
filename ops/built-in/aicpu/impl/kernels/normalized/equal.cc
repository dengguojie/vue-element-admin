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
#include "equal.h"

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kInputNum = 2;
const char *kEqual = "Equal";

#define EQUAL_COMPUTE_CASE(DTYPE, TYPE, CTX)                         \
  case (DTYPE): {                                                    \
    uint32_t result = EqualCompute<TYPE>(CTX);                       \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("Equal kernel compute failed [%d].", result); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
}

namespace aicpu {
uint32_t EqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Equal params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    EQUAL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    EQUAL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    EQUAL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    EQUAL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    EQUAL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    EQUAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    EQUAL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    EQUAL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    EQUAL_COMPUTE_CASE(DT_BOOL, bool, ctx)
    EQUAL_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    EQUAL_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Equal kernel data type [%u] not support.", data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t EqualCpuKernel::EqualCompute(CpuKernelContext &ctx) {
  CalcInfo calc_info;
  calc_info.input_0 = ctx.Input(0);
  calc_info.input_1 = ctx.Input(1);
  calc_info.output = ctx.Output(0);
  DataType input0_type = calc_info.input_0->GetDataType();
  DataType input1_type = calc_info.input_1->GetDataType();
  DataType output_type = calc_info.output->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "DataType of x1 [%d] should be same as x2 [%d].", 
                     input0_type, input1_type)
  KERNEL_LOG_INFO(
      "EqualCpuKernel[%s], input x1 : addr[%p], size[%llu];"
      "input x2: addr[%p], size[%llu];"
      "output: addr[%p], size[%llu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetData(),
      calc_info.input_0->GetDataSize(), calc_info.input_1->GetData(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetData(),
      calc_info.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calc_info),
                      "Generate broadcast info failed.")
  (void)bcast.BCastIndexes(calc_info.x_indexes, calc_info.y_indexes);
  (void)bcast.GetBcastVec(calc_info);

  return EqualCalculate<T>(ctx, calc_info);
}

template <typename T>
uint32_t EqualCpuKernel::EqualCalculate(CpuKernelContext &ctx,
                                      CalcInfo &calc_info) {
  auto input_x1 = reinterpret_cast<T *>(calc_info.input_0->GetData());
  auto input_x2 = reinterpret_cast<T *>(calc_info.input_1->GetData());
  auto output_y = reinterpret_cast<bool *>(calc_info.output->GetData());
  KERNEL_CHECK_NULLPTR(input_x1, KERNEL_STATUS_PARAM_INVALID, 
                           "[%s] get input x1 data failed.", kEqual)
  KERNEL_CHECK_NULLPTR(input_x2, KERNEL_STATUS_PARAM_INVALID, 
                           "[%s] get input x2 data failed.", kEqual)
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID, 
                           "[%s] get output data failed.", kEqual)
  size_t data_num = calc_info.x_indexes.size();
  auto shard_equal = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto x_index = input_x1 + calc_info.x_indexes[i];
      auto y_index = input_x2 + calc_info.y_indexes[i];
      output_y[i] = (*x_index == *y_index);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 1, shard_equal),
                      "Equal calculate failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kEqual, EqualCpuKernel);
}  // namespace aicpu