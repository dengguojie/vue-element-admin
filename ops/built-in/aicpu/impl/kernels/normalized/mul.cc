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

#include "mul.h"
#include <algorithm>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr char *kMul = "Mul";

#define MUL_COMPUTE_CASE(DTYPE, TYPE, CTX)                \
  case (DTYPE): {                                         \
    if (MulCompute<TYPE>(CTX) != KERNEL_STATUS_OK) {      \
      KERNEL_LOG_ERROR("[%s] compute failed.",            \
                       CTX.GetOpType().c_str());          \
      return KERNEL_STATUS_PARAM_INVALID;                 \
    }                                                     \
    break;                                                \
  }

#define MUL_CALCULATE_CASE(RANK, T)                       \
  case (RANK): {                                          \
    if (AlignedCheck(calc_info)) {                        \
      MulCalculate<RANK, T, Eigen::Aligned>(calc_info);   \
    } else {                                              \
      MulCalculate<RANK, T, Eigen::Unaligned>(calc_info); \
    }                                                     \
    break;                                                \
  }
}

namespace aicpu {
uint32_t MulCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[0] failed", ctx.GetOpType().c_str());
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(input_1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get input[1] failed", ctx.GetOpType().c_str());
  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.",
                    ctx.GetOpType().c_str(), input_0->GetDataSize(),
                    input_1->GetDataSize());
    return KERNEL_STATUS_OK;
  }
  // choose compute function depend on dataType
  auto data_type =
      static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    MUL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    MUL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    MUL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    MUL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    MUL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    MUL_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    MUL_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    MUL_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    MUL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    MUL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MUL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MulCpuKernel::MulCompute(CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 0 data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 1 data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get output data failed", ctx.GetOpType().c_str())
  KERNEL_LOG_INFO(
      "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], output "
      "data size is [%llu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetDataSize());
  // broadcast input
  Bcast bcast;
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  (void)bcast.GetBcastVec(calc_info);

 // choose eigen calculate function depend on rank of input
  switch (static_cast<int32_t>(calc_info.shape_out.size())) {
    case 0: {
      T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
      T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
      T *value_out = reinterpret_cast<T *>(calc_info.output->GetData());
      *(value_out) = v0 * v1;
      break;
    }
    MUL_CALCULATE_CASE(1, T)
    MUL_CALCULATE_CASE(2, T)
    MUL_CALCULATE_CASE(3, T)
    MUL_CALCULATE_CASE(4, T)
    MUL_CALCULATE_CASE(5, T)
    MUL_CALCULATE_CASE(6, T)
    default:
      KERNEL_LOG_ERROR("[%s] Rank of output should less than 6 but get [%zu].",
                       ctx.GetOpType().c_str(), calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

bool MulCpuKernel::AlignedCheck(const BCalcInfo &calc_info) {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
void MulCpuKernel::MulCalculate(BCalcInfo &calc_info) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input_0(
      static_cast<T *>(calc_info.input_0->GetData()),
      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input_1(
      static_cast<T *>(calc_info.input_1->GetData()),
      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(
      static_cast<T *>(calc_info.output->GetData()),
      calc_info.output->GetTensorShape()->NumElements());

  Eigen::ThreadPool thread_pool(kThreadNum);
  Eigen::ThreadPoolDevice thread_pool_device(&thread_pool, kThreadNum);

  if (calc_info.input_0->GetTensorShape()->GetDimSizes().empty()) {
    T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
    output.device(thread_pool_device) = v0 * input_1;
    return;
  }

  if (calc_info.input_1->GetTensorShape()->GetDimSizes().empty()) {
    T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
    output.device(thread_pool_device) = input_0 * v1;
    return;
  }

  std::reverse(calc_info.reshape_0.begin(), calc_info.reshape_0.end());
  std::reverse(calc_info.reshape_1.begin(), calc_info.reshape_1.end());
  std::reverse(calc_info.shape_out.begin(), calc_info.shape_out.end());
  std::reverse(calc_info.bcast_0.begin(), calc_info.bcast_0.end());
  std::reverse(calc_info.bcast_1.begin(), calc_info.bcast_1.end());

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_1;

  for (int32_t i = 0; i < RANK; i++) {
    reshape_0[i] = calc_info.reshape_0[i];
    reshape_1[i] = calc_info.reshape_1[i];
    shape_out[i] = calc_info.shape_out[i];
    bcast_0[i] = calc_info.bcast_0[i];
    bcast_1[i] = calc_info.bcast_1[i];
  }

  output.reshape(shape_out).device(thread_pool_device) =
      input_0.reshape(reshape_0).broadcast(bcast_0) *
      input_1.reshape(reshape_1).broadcast(bcast_1);
}

REGISTER_CPU_KERNEL(kMul, MulCpuKernel);
}  // namespace aicpu
