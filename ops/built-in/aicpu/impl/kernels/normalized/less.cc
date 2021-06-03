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

#include "less.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kLess = "Less";

#define LESS_COMPUTE_CASE(DTYPE, TYPE, CTX, CALCINFO)    \
  case (DTYPE): {                                        \
    uint32_t result = LessCompute<TYPE>(CTX, CALCINFO);  \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Less kernel compute failed.");   \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
}

namespace aicpu {
uint32_t LessCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Less check input and output number failed.");
  BCalcInfo calc_info;
  KERNEL_HANDLE_ERROR(LessCheckAndBroadCast(ctx, calc_info),
                      "Less check params or bcast failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LESS_COMPUTE_CASE(DT_INT8, int8_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_INT16, int16_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_INT32, int32_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_INT64, int64_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_UINT8, uint8_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_UINT16, uint16_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_UINT32, uint32_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_UINT64, uint64_t, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_FLOAT, float, ctx, calc_info)
    LESS_COMPUTE_CASE(DT_DOUBLE, double, ctx, calc_info)
    default:
      KERNEL_LOG_ERROR("Less kernel data type [%s] not support.", DTypeStr(data_type));
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t LessCpuKernel::LessCheckAndBroadCast(CpuKernelContext &ctx,
                                              BCalcInfo &calc_info) {
  calc_info.input_0 = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(calc_info.input_0, KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 failed.")
  calc_info.input_1 = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(calc_info.input_1, KERNEL_STATUS_PARAM_INVALID,
                       "Get input 1 failed.")
  calc_info.output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(calc_info.output, KERNEL_STATUS_PARAM_INVALID,
                       "Get output failed.")
  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  DataType input0_type = calc_info.input_0->GetDataType();
  DataType input1_type = calc_info.input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].", DTypeStr(input0_type), DTypeStr(input1_type))
  KERNEL_LOG_DEBUG(
      "LessCpuKernel[%s], input0: size[%llu];"
      "input1: size[%llu], output: size[%llu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calc_info),
                      "Generate broadcast info failed.")
  (void)bcast.BCastIndexes(calc_info.x_indexes, calc_info.y_indexes);
  (void)bcast.GetBcastVec(calc_info);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LessCpuKernel::LessCompute(CpuKernelContext &ctx,
                                    BCalcInfo &calc_info) {
  auto input_x1 = reinterpret_cast<T *>(calc_info.input_0->GetData());
  auto input_x2 = reinterpret_cast<T *>(calc_info.input_1->GetData());
  auto output_y = reinterpret_cast<bool *>(calc_info.output->GetData());

  size_t data_num = calc_info.x_indexes.size();
  uint32_t min_core_num = 1;
  int64_t max_core_num =
      std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }
  auto shard_less = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto x_index = input_x1 + calc_info.x_indexes[i]; // i-th value of input0
      auto y_index = input_x2 + calc_info.y_indexes[i]; // i-th value of input1
      *(output_y + i) = *x_index < *y_index ? true : false;
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_less),
                      "Less Compute failed.")
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLess, LessCpuKernel);
}  // namespace aicpu