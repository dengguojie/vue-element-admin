/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "addcmul.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kAddcmul = "Addcmul";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define ADDCMUL_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = AddcmulCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Addcmul kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }

#define COMPUTE_CASE(TYPE, INDEX, END, INPUT0, INPUT1, INPUT2, INPUT3, OUTPUT) \
  case (TYPE): {                                                               \
    for (; (INDEX) < (END); ++(INDEX)) {                                       \
      *((OUTPUT) + (INDEX)) = *(INPUT0) + *(INPUT3) * *(INPUT1) * *(INPUT2);   \
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
uint32_t AddcmulCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Addcmul check input and output number failed.");
  KERNEL_HANDLE_ERROR(AddcmulParamCheck(ctx), "Addcmul check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ADDCMUL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ADDCMUL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ADDCMUL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    ADDCMUL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    ADDCMUL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    ADDCMUL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    ADDCMUL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      KERNEL_LOG_ERROR("Addcmul kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t AddcmulCpuKernel::AddcmulParamCheck(CpuKernelContext &ctx) {
  // the non null of input_data, x1, x2, value, y has been verified in
  // NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *input_2 = ctx.Input(2);
  Tensor *input_3 = ctx.Input(3);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  DataType input2_type = input_2->GetDataType();
  DataType input3_type = input_3->GetDataType();
  KERNEL_CHECK_FALSE(
      ((input0_type == input1_type) && (input0_type == input2_type) &&
       (input0_type == input3_type)),
      KERNEL_STATUS_PARAM_INVALID,
      "The data type of input0 [%s] need be same with "
      "input1 [%s], input2 [%s] and input3 [%s].",
      DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str(),
      DTypeStr(input2_type).c_str(), DTypeStr(input3_type).c_str())
  KERNEL_LOG_DEBUG(
      "AddcmulCpuKernel[%s], input0: size[%llu]; input1: size[%llu];"
      "input2: size[%llu]; input3: size[%llu], output: size[%llu].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(),
      input_2->GetDataSize(), input_3->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

/**
 * special compute is used in the following situations.
 * 1. the shapes of input_data and the result of x1 mumtiply x2 are the same
 * 2. input_data is a 1D tensor with only one element or input_data is scalar
 * 3. result of x1 mumtiply x2 is a 1D tensor with only one element or it is
 * scalar
 */
template <typename T>
void AddcmulCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                      int64_t end, CpuKernelContext &ctx,
                                      CalcInfo &mul_info) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto input3 = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t idx = start;
  T mul_result = *input3 * *input1 * *input2;
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      ADD_NO_BCAST_COMPUTE(idx, end, input0 + idx, input1, input2, input3,
                           output, mul_info)
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      ADD_NO_BCAST_COMPUTE(idx, end, input0, input1, input2, input3, output,
                           mul_info)
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input0 + i) + mul_result;
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
void AddcmulCpuKernel::AddBcastCompute(Bcast &add_bcast, int64_t start,
                                       int64_t end, CpuKernelContext &ctx,
                                       CalcInfo &mul_info) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto input3 = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t idx = start;
  Bcast *mul_bcast = mul_info.bcast;
  switch (mul_info.type) {
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
                 input1 + mul_bcast->GetBroadcastXIndex(
                              add_bcast.GetBroadcastYIndex(idx)),
                 input2 + mul_bcast->GetBroadcastYIndex(
                              add_bcast.GetBroadcastYIndex(idx)),
                 input3, output)
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(mul_info.type));
      break;
  }
}

template <typename T>
uint32_t AddcmulCpuKernel::NoBcastCompute(CpuKernelContext &ctx,
                                          CalcInfo &mul_info) {
  int64_t input0_elements_nums = ctx.Input(0)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
      input0_elements_nums == mul_info.elements_nums
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

    auto sharder_addcmul = [&](int64_t start, int64_t end) {
      SpecialCompute<T>(type, start, end, ctx, mul_info);
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_addcmul),
        "Addcmul Compute failed.")
  } else {
    SpecialCompute<T>(type, 0, data_num, ctx, mul_info);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AddcmulCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast,
                                        CalcInfo &mul_info) {
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    auto sharder_addcmul = [&](int64_t start, int64_t end) {
      AddBcastCompute<T>(bcast, start, end, ctx, mul_info);
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_addcmul),
        "Addcmul Compute failed.")
  } else {
    AddBcastCompute<T>(bcast, 0, data_num, ctx, mul_info);
  }
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t AddcmulCpuKernel::AddcmulCompute(CpuKernelContext &ctx) {
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

  bool isNeedBcastMul = (input1_shape != input2_shape) &&
                        (input1_elements_nums != 1) &&
                        (input2_elements_nums != 1);
  std::vector<int64_t> mul_result_shape;
  Bcast *bcastMul = nullptr;
  if (isNeedBcastMul) {
    bcastMul = new Bcast(input1_shape, input2_shape);
    if (!bcastMul->IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    BCalcInfo calc_info;
    calc_info.input_0 = ctx.Input(1);
    calc_info.input_1 = ctx.Input(kThirdInputIndex);
    calc_info.output = ctx.Output(0);
    Bcast bcast;
    bcast.GenerateBcastInfo(calc_info);
    (void)bcast.GetBcastVec(calc_info);
    mul_result_shape = calc_info.shape_out;
  } else {
    mul_result_shape = input1_elements_nums == 1 ? input2_shape : input1_shape;
  }
  int64_t mul_result_elements_nums = 1;
  for (uint i = 0; i < mul_result_shape.size(); i++) {
    mul_result_elements_nums = mul_result_elements_nums * mul_result_shape[i];
  }
  BcastShapeType mul_type =
      input1_shape == input2_shape
          ? BcastShapeType::SAME_SHAPE
          : (input1_elements_nums == 1
                 ? BcastShapeType::X_ONE_ELEMENT
                 : (input2_elements_nums == 1 ? BcastShapeType::Y_ONE_ELEMENT
                                              : BcastShapeType::DIFF_SHAPE));
  CalcInfo mul_info;
  mul_info.isNeedBcast = isNeedBcastMul;
  mul_info.bcast = bcastMul;
  mul_info.elements_nums = mul_result_elements_nums;
  mul_info.type = mul_type;

  bool isNeedBcastAdd = (input0_shape != mul_result_shape) &&
                        (input0_elements_nums != 1) &&
                        (mul_result_elements_nums != 1);
  if (isNeedBcastAdd) {
    Bcast bcastAdd(input0_shape, mul_result_shape);
    if (!bcastAdd.IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastCompute<T>(ctx, bcastAdd, mul_info);
  } else {
    return NoBcastCompute<T>(ctx, mul_info);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAddcmul, AddcmulCpuKernel);
}  // namespace aicpu