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
#include "greater_equal.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kInputNum = 2;
constexpr char const *kGreaterEqual = "GreaterEqual";
// when input data size is more than kParallelDataNum, use Parallel func
constexpr int64_t kParallelDataNum = 8 * 1024;
constexpr int64_t kParallelDataNumSameShape = 32 * 1024;

#define GREATER_EQUAL_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                              \
    uint32_t result = GreaterEqualCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("GreaterEqual kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t GreaterEqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(GreaterEqualParamCheck(ctx),
                      "GreaterEqual check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    GREATER_EQUAL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    GREATER_EQUAL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("GreaterEqual kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t GreaterEqualCpuKernel::GreaterEqualParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "GreaterEqual check input and output number failed.");
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  DataType output_type = output->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] should be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(),
                     DTypeStr(input1_type).c_str());
  KERNEL_LOG_DEBUG(
      "GreaterEqualCpuKernel[%s], input0: size[%llu] dtype[%s]; "
      "input1: size[%llu] dtype[%s], output: size[%llu] dtype[%s].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(),
      DTypeStr(input0_type).c_str(), input_1->GetDataSize(),
      DTypeStr(input1_type).c_str(), output->GetDataSize(),
      DTypeStr(output_type).c_str());

  return KERNEL_STATUS_OK;
}

// special compute is used in the following situations.
// 1. the shapes of input1 and input2 are the same
// 2. input1 is a 1D tensor with only one element or input1 is scalar
// 3. input2 is a 1D tensor with only one element or input2 is scalar
// 4. the shapes of input1 and input2 are different
template <typename T>
void GreaterEqualCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                           int64_t end, const T *input1,
                                           const T *input2, bool *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        output[i] = input1[i] >= input2[i];
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        output[i] = *input1 >= input2[i];
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        output[i] = input1[i] >= *input2;
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
uint32_t GreaterEqualCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  const int64_t input0_elements_nums = ctx.Input(0)->NumElements();
  const int64_t input1_elements_nums = ctx.Input(1)->NumElements();
  const int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = (input0_elements_nums == input1_elements_nums) ?
      BcastShapeType::SAME_SHAPE :
      ((input0_elements_nums == 1) ?
      BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    const int64_t min_core_num = 4;
    const int64_t max_core_num = std::max(
        min_core_num,
        static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum));
    const int64_t per_unit_size = data_num / std::min(data_num, max_core_num);

    auto sharder_less = [&](int64_t start, int64_t end) {
      SpecialCompute<T>(type, start, end, input0, input1, output);
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharder_less),
        "GreaterEqual Compute failed.")
  } else {
    SpecialCompute<T>(type, 0, data_num, input0, input1, output);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GreaterEqualCpuKernel::BcastCompute(CpuKernelContext &ctx,
                                             Bcast &bcast) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());

  const int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    const int64_t min_core_num = 4;
    const int64_t max_core_num = std::max(
        min_core_num,
        static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum));
    const int64_t per_unit_size = data_num / std::min(data_num, max_core_num);

    auto sharder_less = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        output[i] = input0[bcast.GetBroadcastXIndex(i)] >=
                    input1[bcast.GetBroadcastYIndex(i)];
      }
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharder_less),
        "GreaterEqual Compute failed.")
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      output[i] = input0[bcast.GetBroadcastXIndex(i)] >=
                  input1[bcast.GetBroadcastYIndex(i)];
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GreaterEqualCpuKernel::GreaterEqualCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  // choose generate broadcast information or not
  const bool is_need_call_bcast =
      !((input0_shape == input1_shape) || (input0_elements_nums == 1) ||
        (input1_elements_nums == 1));
  if (is_need_call_bcast) {
    Bcast bcast(input0_shape, input1_shape);
    KERNEL_CHECK_FALSE(bcast.IsValid(), KERNEL_STATUS_PARAM_INVALID,
                       "GreaterEqual broadcast failed.");
    return BcastCompute<T>(ctx, bcast);
  } else {
    return NoBcastCompute<T>(ctx);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kGreaterEqual, GreaterEqualCpuKernel);
}  // namespace aicpu
