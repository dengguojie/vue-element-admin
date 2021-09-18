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
#include "add.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const char *kAdd = "Add";
}
namespace aicpu {
uint32_t AddCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }

  // choose compute function depend on dataType
  auto data_type =
      static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return AddCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return AddCompute<float>(ctx);
    case DT_DOUBLE:
      return AddCompute<double>(ctx);
    case DT_INT8:
      return AddCompute<int8_t>(ctx);
    case DT_INT16:
      return AddCompute<int16_t>(ctx);
    case DT_INT32:
      return AddCompute<int32_t>(ctx);
    case DT_INT64:
      return AddCompute<int64_t>(ctx);
    case DT_UINT8:
      return AddCompute<uint8_t>(ctx);
    case DT_UINT16:
      return AddCompute<uint16_t>(ctx);
    case DT_UINT32:
      return AddCompute<uint32_t>(ctx);
    case DT_UINT64:
      return AddCompute<uint64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t AddCpuKernel::AddCompute(CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);

  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[0] data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[1] data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get output data failed", ctx.GetOpType().c_str())

  KERNEL_LOG_INFO(
      "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], "
      "output data size is [%llu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetDataSize());
  
  // broadcast input
  Bcast bcast;
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  (void)bcast.GetBcastVec(calc_info);
  int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
  switch (rank) {
    case 0:
    {
      T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
      T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
      T *value_out = reinterpret_cast<T *>(calc_info.output->GetData());
      *(value_out) = v0 + v1;
      return KERNEL_STATUS_OK;
    }
    case 1:
      return AddCalculateWithAlignedCheck<1, T>(ctx, calc_info);
    case 2:
      return AddCalculateWithAlignedCheck<2, T>(ctx, calc_info);
    case 3:
      return AddCalculateWithAlignedCheck<3, T>(ctx, calc_info);
    case 4:
      return AddCalculateWithAlignedCheck<4, T>(ctx, calc_info);
    case 5:
      return AddCalculateWithAlignedCheck<5, T>(ctx, calc_info);
    case 6:
      return AddCalculateWithAlignedCheck<6, T>(ctx, calc_info);
    case 7:
      return AddCalculateWithAlignedCheck<7, T>(ctx, calc_info);
    case 8:
      return AddCalculateWithAlignedCheck<8, T>(ctx, calc_info);
    default:
      KERNEL_LOG_ERROR("[%s] Rank of output should less than 8 but get [%zu].",
                       ctx.GetOpType().c_str(), calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <int32_t RANK, typename T>
uint32_t AddCpuKernel::AddCalculateWithAlignedCheck(const CpuKernelContext &ctx, BCalcInfo &calc_info) {
  if (AlignedCheck(calc_info)) {
    return AddCalculate<RANK, T, Eigen::Aligned>(ctx, calc_info);
  }
  return AddCalculate<RANK, T, Eigen::Unaligned>(ctx, calc_info);
}

bool AddCpuKernel::AlignedCheck(const BCalcInfo &calc_info) {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t AddCpuKernel::AddCalculate(const CpuKernelContext &ctx, BCalcInfo &calc_info) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(
      static_cast<T *>(calc_info.input_0->GetData()),
      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(
      static_cast<T *>(calc_info.input_1->GetData()),
      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(
      static_cast<T *>(calc_info.output->GetData()),
      calc_info.output->GetTensorShape()->NumElements());
  auto input_shape_0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
  auto input_shape_1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
  if (input_shape_0.empty()) {
    T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
    output = v0 + input1;
    return KERNEL_STATUS_OK;
  }

  if (input_shape_1.empty()) {
    T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
    output = input0 + v1;
    return KERNEL_STATUS_OK;
  }

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_1;

  for (int32_t i = 0; i < RANK; i++) {
    reshape_0[RANK - i - 1] = calc_info.reshape_0[i];
    reshape_1[RANK - i - 1] = calc_info.reshape_1[i];
    shape_out[RANK - i - 1] = calc_info.shape_out[i];
    bcast_0[RANK - i - 1] = calc_info.bcast_0[i];
    bcast_1[RANK - i - 1] = calc_info.bcast_1[i];
  }
  output.reshape(shape_out) =
      input0.reshape(reshape_0).broadcast(bcast_0) + input1.reshape(reshape_1).broadcast(bcast_1);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdd, AddCpuKernel);
}  // namespace aicpu
