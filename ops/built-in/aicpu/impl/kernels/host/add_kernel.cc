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

#include "add_kernel.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char* ADD = "Add";

#define ADD_COMPUTE_CASE(DTYPE, TYPE, CTX)              \
  case (DTYPE): {                                       \
    if (AddCompute<TYPE>(CTX) != KERNEL_STATUS_OK) {    \
      KERNEL_LOG_ERROR("Add kernel compute failed.");   \
      return KERNEL_STATUS_PARAM_INVALID;               \
    }                                                   \
    break;                                              \
  }

#define ADD_CALCULATE_CASE(RANK, T)     \
  case (RANK): {                        \
    AddCalculate<RANK, T>(calc_info);   \
    break;                              \
  }
}

namespace aicpu {
uint32_t AddKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Add folding kernel in.");
  if (NormalCheck(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Check add %s failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    ADD_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    ADD_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    ADD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    ADD_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    ADD_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    ADD_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    ADD_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    ADD_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    ADD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ADD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ADD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Add kernel data type %u not support.", data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("Add kernel run success.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AddKernel::AddCompute(CpuKernelContext &ctx) {
  CalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed")
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed")
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_LOG_INFO("Add %s, input0: addr=%p, size=%llu; input1: addr=%p, size=%llu; output: addr %p, size %llu.",
                  ctx.GetOpType().c_str(),
                  calc_info.input_0->GetData(), calc_info.input_0->GetDataSize(),
                  calc_info.input_1->GetData(), calc_info.input_1->GetDataSize(),
                  calc_info.output->GetData(), calc_info.output->GetDataSize());

  Bcast bcast;
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Generate broadcast info failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  (void)bcast.GetBcastVec(calc_info);

  switch (static_cast<int32_t>(calc_info.shape_out.size())) {
    case 0: {
      T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
      T v1 = *(reinterpret_cast<const T *>(calc_info.input_1->GetData()));
      T *value_out = reinterpret_cast<T *>(calc_info.output->GetData());
      *(value_out) = v0 + v1;
      break;
    }
    ADD_CALCULATE_CASE(1, T)
    ADD_CALCULATE_CASE(2, T)
    ADD_CALCULATE_CASE(3, T)
    ADD_CALCULATE_CASE(4, T)
    ADD_CALCULATE_CASE(5, T)
    ADD_CALCULATE_CASE(6, T)
    default:
      KERNEL_LOG_ERROR("Add kernel not support rank=%zu.", calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <int32_t RANK, typename T>
void AddKernel::AddCalculate(CalcInfo &calc_info) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>> eigen_input_0(static_cast<T *>(calc_info.input_0->GetData()),
                                                      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>> eigen_input_1(static_cast<T *>(calc_info.input_1->GetData()),
                                                      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>> eigen_output(static_cast<T *>(calc_info.output->GetData()),
                                                     calc_info.output->GetTensorShape()->NumElements());

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_1;
  for (int32_t i = 0; i < RANK; i++){
    reshape_0[i] = calc_info.reshape_0[i];
    reshape_1[i] = calc_info.reshape_1[i];
    shape_out[i] = calc_info.shape_out[i];
    bcast_0[i] = calc_info.bcast_0[i];
    bcast_1[i] = calc_info.bcast_1[i];
  }

  Eigen::ThreadPool thread_pool(kThreadNum);
  Eigen::ThreadPoolDevice thread_pool_device(&thread_pool, kThreadNum);
  eigen_output.reshape(shape_out).device(thread_pool_device) =
          eigen_input_0.reshape(reshape_0).broadcast(bcast_0) + eigen_input_1.reshape(reshape_1).broadcast(bcast_1);
}

REGISTER_CPU_KERNEL(ADD, AddKernel);
} // namespace aicpu
