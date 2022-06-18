/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unaddc_div required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "addcdiv.h"

#include <math.h>
#include "bcast.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
const char *kAddcdiv = "Addcdiv";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define ADDCDIV_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = AddcdivCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Addcdiv kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }

#define COMPUTE_CASE(TYPE, INDEX, END, INPUT0, INPUT1, INPUT2, INPUT3, OUTPUT) \
  case (TYPE): {                                                               \
    for (; (INDEX) < (END); ++(INDEX)) {                                       \
      *((OUTPUT) + (INDEX)) = *(INPUT0) + *(INPUT3) * (*(INPUT1) / *(INPUT2)); \
    }                                                                          \
    break;                                                                     \
  }

#define ADD_NO_BCAST_COMPUTE(INDEX, END, INPUT0, INPUT1, INPUT2, INPUT3,    \
                             OUTPUT, CALC_INFO)                             \
  switch ((CALC_INFO).type) {                                               \
    COMPUTE_CASE(BcastShapeType::SAME_SHAPE, INDEX, END, INPUT0,            \
                 (INPUT1) + (INDEX), (INPUT2) + (INDEX), INPUT3, OUTPUT)    \
    COMPUTE_CASE(BcastShapeType::X_ONE_ELEMENT, INDEX, END, INPUT0, INPUT1, \
                 (INPUT2) + (INDEX), INPUT3, OUTPUT)                        \
    COMPUTE_CASE(BcastShapeType::Y_ONE_ELEMENT, INDEX, END, INPUT0,         \
                 (INPUT1) + (INDEX), INPUT2, INPUT3, OUTPUT)                \
    COMPUTE_CASE(BcastShapeType::DIFF_SHAPE, INDEX, END, INPUT0,            \
                 (INPUT1) + (CALC_INFO).bcast->GetBroadcastXIndex(INDEX),   \
                 (INPUT2) + (CALC_INFO).bcast->GetBroadcastYIndex(INDEX),   \
                 INPUT3, OUTPUT)                                            \
    default:                                                                \
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));     \
      break;                                                                \
  }
}  // namespace

namespace aicpu {
uint32_t AddcdivCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Addcdiv check input and output number failed.");
  KERNEL_HANDLE_ERROR(AddcdivParamCheck(ctx), "Addcdiv check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ADDCDIV_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ADDCDIV_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    ADDCDIV_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ADDCDIV_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      KERNEL_LOG_ERROR("Addcdiv kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t AddcdivCpuKernel::AddcdivParamCheck(CpuKernelContext &ctx) {
  // the non null of input_data, x1, x2, value, y has been verified in
  // NormalCheck
  Tensor *input0 = ctx.Input(0);
  Tensor *input1 = ctx.Input(1);
  Tensor *input2 = ctx.Input(2);
  Tensor *input3 = ctx.Input(3);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input0->GetDataType();
  DataType input1_type = input1->GetDataType();
  DataType input2_type = input2->GetDataType();
  KERNEL_CHECK_FALSE(
      ((input0_type == input1_type) && (input0_type == input2_type)),
      KERNEL_STATUS_PARAM_INVALID,
      "The data type of input0 [%s] need be same with "
      "input1 [%s] and input2 [%s].",
      DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str(),
      DTypeStr(input2_type).c_str())
  KERNEL_LOG_DEBUG(
      "AddcdivCpuKernel[%s], input0: size[%llu]; input1: size[%llu];"
      "input2: size[%llu]; input3: size[%llu], output: size[%llu].",
      ctx.GetOpType().c_str(), input0->GetDataSize(), input1->GetDataSize(),
      input2->GetDataSize(), input3->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

/*
special compute is used in the following situations.
1. the shapes of input1 and input2 are the same.
2. input1 is a 1D tensor with only one element or input1 is scalar.
3. input2 is a 1D tensor with only one element or input2 is scalar.
*/

template <typename T>
void AddcdivCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                      int64_t end, CpuKernelContext &ctx,
                                      CalcInfo &div_info) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto input3 = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t idx = start;
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      ADD_NO_BCAST_COMPUTE(idx, end, input0 + idx, input1, input2, input3,
                           output, div_info)
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      ADD_NO_BCAST_COMPUTE(idx, end, input0, input1, input2, input3, output,
                           div_info)
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input0 + i) + *input3 * (*input1 / *input2);
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
void AddcdivCpuKernel::AddBcastCompute(Bcast &add_bcast, int64_t start,
                                       int64_t end, CpuKernelContext &ctx,
                                       CalcInfo &div_info) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto input3 = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t idx = start;
  Bcast *div_bcast = div_info.bcast;
  switch (div_info.type) {
    COMPUTE_CASE(BcastShapeType::SAME_SHAPE, idx, end,
                 input0 + add_bcast.GetBroadcastXIndex(idx),
                 input1 + add_bcast.GetBroadcastYIndex(idx),
                 input2 + add_bcast.GetBroadcastYIndex(idx), input3, output)
    COMPUTE_CASE(BcastShapeType::X_ONE_ELEMENT, idx, end,
                 input0 + add_bcast.GetBroadcastXIndex(idx), input1,
                 input2 + add_bcast.GetBroadcastYIndex(idx), input3, output)
    COMPUTE_CASE(BcastShapeType::Y_ONE_ELEMENT, idx, end,
                 input0 + add_bcast.GetBroadcastXIndex(idx),
                 input1 + add_bcast.GetBroadcastYIndex(idx), input2, input3,
                 output)
    COMPUTE_CASE(BcastShapeType::DIFF_SHAPE, idx, end,
                 input0 + add_bcast.GetBroadcastXIndex(idx),
                 input1 + div_bcast->GetBroadcastXIndex(
                              add_bcast.GetBroadcastYIndex(idx)),
                 input2 + div_bcast->GetBroadcastYIndex(
                              add_bcast.GetBroadcastYIndex(idx)),
                 input3, output)
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(div_info.type));
      break;
  }
}

template <typename T>
uint32_t AddcdivCpuKernel::NoBcastCompute(CpuKernelContext &ctx,
                                          CalcInfo &div_info) {
  int64_t input0_elements_nums = ctx.Input(0)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
      input0_elements_nums == div_info.elements_nums
          ? BcastShapeType::SAME_SHAPE
          : (input0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT
                                       : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_addcdiv = [&](int64_t start, int64_t end) {
      SpecialCompute<T>(type, start, end, ctx, div_info);
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_addcdiv),
        "Addcdiv Compute failed.")
  } else {
    SpecialCompute<T>(type, 0, data_num, ctx, div_info);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AddcdivCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast,
                                        CalcInfo &div_info) {
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_addcdiv = [&](int64_t start, int64_t end) {
      AddBcastCompute<T>(bcast, start, end, ctx, div_info);
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_addcdiv),
        "Addcdiv Compute failed.")
  } else {
    AddBcastCompute<T>(bcast, 0, data_num, ctx, div_info);
  }
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t AddcdivCpuKernel::AddcdivCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  std::vector<int64_t> input0_shape =
      input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  std::vector<int64_t> input1_shape =
      input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  Tensor *input2_tensor = ctx.Input(2);
  std::vector<int64_t> input2_shape =
      input2_tensor->GetTensorShape()->GetDimSizes();
  int64_t input2_elements_nums = input2_tensor->NumElements();
  // input2 cannot be zero
  auto input2_data = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  float f_value = 0.0;
  for (int64_t i = 0; i < input2_elements_nums; i++) {
    if ((*input2_data) == static_cast<T>(f_value)) {
      KERNEL_LOG_ERROR("The dividend of Addcdiv cannot be 0.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    input2_data++;
  }

  bool isNeedBcastDiv = (input1_shape == input2_shape) ||
                        (input1_elements_nums == 1) ||
                        (input2_elements_nums == 1);
  std::vector<int64_t> div_result_shape;
  Bcast *bcastDiv = new Bcast(input1_shape, input2_shape);
  if (isNeedBcastDiv) {
    div_result_shape = input1_elements_nums == 1 ? input2_shape : input1_shape;
  } else {
    if (!bcastDiv->IsValid()) {
      KERNEL_LOG_ERROR("The division broadcast in the [%s] operation failed",
                       ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    BCalcInfo calc_info;
    calc_info.input_0 = ctx.Input(1);
    calc_info.input_1 = ctx.Input(kThirdInputIndex);
    calc_info.output = ctx.Output(0);
    Bcast bcast;
    bcast.GenerateBcastInfo(calc_info);
    (void)bcast.GetBcastVec(calc_info);
    div_result_shape = calc_info.shape_out;
  }
  int64_t div_result_elements_nums = 1;
  for (uint i = 0; i < div_result_shape.size(); i++) {
    div_result_elements_nums = div_result_elements_nums * div_result_shape[i];
  }
  BcastShapeType div_type =
      input1_shape == input2_shape
          ? BcastShapeType::SAME_SHAPE
          : (input1_elements_nums == 1
                 ? BcastShapeType::X_ONE_ELEMENT
                 : (input2_elements_nums == 1 ? BcastShapeType::Y_ONE_ELEMENT
                                              : BcastShapeType::DIFF_SHAPE));
  CalcInfo div_info;
  div_info.isNeedBcast = isNeedBcastDiv;
  div_info.bcast = bcastDiv;
  div_info.elements_nums = div_result_elements_nums;
  div_info.type = div_type;

  bool isNeedBcastAdd = (input0_shape == div_result_shape) ||
                        (input0_elements_nums == 1) ||
                        (div_result_elements_nums == 1);
  if (isNeedBcastAdd) {
    return NoBcastCompute<T>(ctx, div_info);
  } else {
    Bcast bcastAdd(input0_shape, div_result_shape);
    if (!bcastAdd.IsValid()) {
      KERNEL_LOG_ERROR("The addition broadcast in the [%s] operation failed",
                       ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastCompute<T>(ctx, bcastAdd, div_info);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAddcdiv, AddcdivCpuKernel);
}  // namespace aicpu
