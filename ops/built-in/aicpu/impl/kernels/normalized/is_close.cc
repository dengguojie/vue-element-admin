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

#include "is_close.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const float rtol_ = 1e-05;
const float atol_ = 1e-08;
const uint32_t CPUNUM = 2;
const char *kIsClose = "IsClose";

const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define ISCLOSE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = IsCloseCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("IsClose kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t IsCloseCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "IsClose check input and output number failed.");
  KERNEL_HANDLE_ERROR(IsCloseParamCheck(ctx), "IsClose check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ISCLOSE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    ISCLOSE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ISCLOSE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ISCLOSE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("IsClose kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t IsCloseCpuKernel::IsCloseParamCheck(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(),
                     DTypeStr(input1_type).c_str())
  KERNEL_LOG_DEBUG(
      "IsCloseCpuKernel[%s], input0: size[%llu];"
      "input1: size[%llu], output: size[%llu].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(),
      output->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IsCloseCpuKernel::IsCloseCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  bool noNeedBcast = (input0_shape == input1_shape) ||
                     (input0_elements_nums == 1) ||
                     (input1_elements_nums == 1);
  if (noNeedBcast) {
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

template <typename T>
uint32_t IsCloseCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t input0_elements_nums = ctx.Input(0)->NumElements();
  int64_t input1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
      input0_elements_nums == input1_elements_nums
          ? BcastShapeType::SAME_SHAPE
          : (input0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT
          : BcastShapeType::Y_ONE_ELEMENT);
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - CPUNUM);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_is_close = [&](int64_t start, int64_t end) {
      SpecialCompute<T>(type, start, end, input0, input1, output);
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_is_close),
        "TensorEqual Compute failed.")
  } else {
    SpecialCompute<T>(type, 0, data_num, input0, input1, output);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IsCloseCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - CPUNUM);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_is_close = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = Eigen::numext::abs(*(input0 + bcast.GetBroadcastXIndex(i)) -
                                      *(input1 + bcast.GetBroadcastYIndex(i))) <=
                             (static_cast<T>(rtol_) -
                              static_cast<T>(atol_) *
                                  Eigen::numext::abs(
                                      *(input1 + bcast.GetBroadcastYIndex(i))))
                         ? true
                         : false;
      }
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_is_close),
        "IsClose Compute failed.")
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      *(output + i) =
          Eigen::numext::abs(*(input0 + bcast.GetBroadcastXIndex(i)) -
                             *(input1 + bcast.GetBroadcastYIndex(i))) <=
                  (static_cast<T>(rtol_) -
                   static_cast<T>(atol_) *
                      Eigen::numext::abs(*(input1 + bcast.GetBroadcastYIndex(i))))
              ? true
              : false;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void IsCloseCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                      int64_t end, const T *input1,
                                      const T *input2, bool *output) {
  switch (type) {
    case  BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) =
            (Eigen::numext::abs(*(input1 + i) - *(input2 + i))) <=
            (static_cast<T>(rtol_) -
             static_cast<T>(atol_) * Eigen::numext::abs(*(input2 + i)));
      }
      break;
    case  BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) =
            (Eigen::numext::abs(*input1 - *(input2 + i))) <=
            (static_cast<T>(rtol_) -
             static_cast<T>(atol_) * Eigen::numext::abs(*(input2 + i)));
      }
      break;
    case  BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = (Eigen::numext::abs(*(input1 + i) - *input2)) <=
                        (static_cast<T>(rtol_) -
                        static_cast<T>(atol_) * (Eigen::numext::abs(*input2)));
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

REGISTER_CPU_KERNEL(kIsClose, IsCloseCpuKernel);
}  // namespace aicpu
