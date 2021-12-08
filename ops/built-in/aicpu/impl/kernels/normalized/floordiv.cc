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
#include "floordiv.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kFloorDiv = "FloorDiv";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 4 * 1024;
const int64_t kParallelDataNumSameShape = 16 * 1024;
const int64_t kParallelDataNumSameShapeMid = 32 * 1024;

#define FLOORDIV_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                          \
    uint32_t result = FloorDivCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("FloorDiv kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t FloorDivCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kFloorDiv);
  KERNEL_HANDLE_ERROR(FloorDivParamCheck(ctx), "[%s] check params failed.",
                      kFloorDiv);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    FLOORDIV_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    FLOORDIV_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    FLOORDIV_COMPUTE_CASE(DT_FLOAT, float, ctx)
    FLOORDIV_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("FloorDiv kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t FloorDivCpuKernel::FloorDivParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(),
                     DTypeStr(input1_type).c_str())

  return KERNEL_STATUS_OK;
}

// special compute is used in the following situations.
// 1. the shapes of input1 and input2 are the same
// 2. input1 is a 1D tensor with only one element or input1 is scalar
// 3. input2 is a 1D tensor with only one element or input2 is scalar
// 4. the shapes of input1 and input2 are different
template <typename T>
uint32_t FloorDivCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                           int64_t end, const T *input1,
                                           const T *input2, T *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        if (*(input2 + i) == static_cast<T>(0)) {
          KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
          return KERNEL_STATUS_INNER_ERROR;
        }
        *(output + i) = Eigen::numext::floor(*(input1 + i) / *(input2 + i));
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        if (*(input2 + i) == static_cast<T>(0)) {
          KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
          return KERNEL_STATUS_INNER_ERROR;
        }
        *(output + i) = Eigen::numext::floor(*input1 / *(input2 + i));
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        if (*input2 == static_cast<T>(0)) {
          KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
          return KERNEL_STATUS_INNER_ERROR;
        }
        *(output + i) = Eigen::numext::floor(*(input1 + i) / *input2);
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FloorDivCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
      in0_elements_nums == in1_elements_nums
          ? BcastShapeType::SAME_SHAPE
          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT
                                    : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    uint32_t status = KERNEL_STATUS_OK;
    auto sharder_floor_div = [&](int64_t start, int64_t end) {
      uint32_t status_sharder =
          SpecialCompute<T>(type, start, end, in0, in1, out);
      if (status_sharder != KERNEL_STATUS_OK) {
        status = status_sharder;
      }
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_floor_div),
        "FloorDiv Compute failed.")
    return status;
  }

  return SpecialCompute<T>(type, 0, data_num, in0, in1, out);
}

template <typename T>
uint32_t FloorDivCpuKernel::BcastParallelCompute(CpuKernelContext &ctx,
                                                 Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  uint32_t min_core_num = 1;
  uint32_t max_core_num = std::max(
      min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num <= kParallelDataNumMid) {
    max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
  }

  if (max_core_num > data_num) {
    max_core_num = data_num;
  }
  uint32_t status = KERNEL_STATUS_OK;
  auto sharder_floor_div = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      if (*(in1 + bcast.GetBroadcastYIndex(i)) == static_cast<T>(0)) {
        KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
        status = KERNEL_STATUS_INNER_ERROR;
        break;
      }
      *(out + i) = Eigen::numext::floor(*(in0 + bcast.GetBroadcastXIndex(i)) /
                                        *(in1 + bcast.GetBroadcastYIndex(i)));
    }
  };

  KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                  sharder_floor_div),
      "FloorDiv Compute failed.")
  return status;
}

template <typename T>
uint32_t FloorDivCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();

  if (data_num >= kParallelDataNum) {
    return BcastParallelCompute<T>(ctx, bcast);
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      if (*(in1 + bcast.GetBroadcastYIndex(i)) == static_cast<T>(0)) {
        KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      *(out + i) = Eigen::numext::floor(*(in0 + bcast.GetBroadcastXIndex(i)) /
                                        *(in1 + bcast.GetBroadcastYIndex(i)));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FloorDivCpuKernel::FloorDivCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  bool isNeedBcast = (input0_shape == input1_shape) ||
                     (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (isNeedBcast) {
    return NoBcastCompute<T>(ctx);
  } else {
    Bcast bcast(input0_shape, input1_shape);
    if (!bcast.IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return BcastCompute<T>(ctx, bcast);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFloorDiv, FloorDivCpuKernel);
}  // namespace aicpu
