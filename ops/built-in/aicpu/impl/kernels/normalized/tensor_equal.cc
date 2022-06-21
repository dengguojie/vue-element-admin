/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tensor_equal.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kTensorEqual = "TensorEqual";
// when input data size is more than k_parallel_data_num, use Parallel func
const int64_t k_parallel_data_num = 2 * 1024;
const int64_t k_parallel_data_num_mid = 16 * 1024;
const int64_t k_parallel_data_num_same_shape = 7 * 1024;
const int64_t k_parallel_data_num_same_shape_mid = 35 * 1024;

#define TEQ_COMPUTE_CASE(DTYPE, TYPE, CTX)                    \
  case (DTYPE): {                                             \
    uint32_t result = TensorEqualCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("TensorEqual kernel compute failed."); \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t TensorEqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "TensorEqual check input and output number failed.");
  KERNEL_HANDLE_ERROR(TensorEqualParamCheck(ctx),
                      "TensorEqual check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    TEQ_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    TEQ_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    TEQ_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TEQ_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    TEQ_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    TEQ_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    TEQ_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    TEQ_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    TEQ_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    TEQ_COMPUTE_CASE(DT_FLOAT, float, ctx)
    TEQ_COMPUTE_CASE(DT_DOUBLE, double, ctx)

    default:
      KERNEL_LOG_ERROR("TensorEqual kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t TensorEqualCpuKernel::TensorEqualParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
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
      "TensorEqualCpuKernel[%s], input0: size[%llu];"
      "input1: size[%llu], output: size[%llu].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(),
      output->GetDataSize());

  return KERNEL_STATUS_OK;
}

/*
 special compute is used in the following situations.
 1. the shapes of input1 and input2 are the same
 2. the shapes of input1 and input2 are different
*/

template <typename T>
void TensorEqualCpuKernel::SpecialCompute(int64_t start, int64_t end,
                                          const T *input1, const T *input2,
                                          bool *output) {
  *output = true;
  for (int64_t i = start; i < end; ++i) {
    if (*(input1 + i) != *(input2 + i)) {
     *output = false;
      break;
    }
  }
}

template <typename T>
uint32_t TensorEqualCpuKernel::TensorEqualCompute(CpuKernelContext &ctx) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t input0_elements_nums = ctx.Input(0)->NumElements();
  int64_t input1_elements_nums = ctx.Input(1)->NumElements();
  auto input0_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input1_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  int64_t data_num = input0_elements_nums;

  if (input0_elements_nums == input1_elements_nums && input0_shape == input1_shape) {
    if (data_num >= k_parallel_data_num_same_shape) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num =
          std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

      if (data_num <= k_parallel_data_num_same_shape_mid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }

      if (max_core_num > data_num) {
        max_core_num = data_num;
      }

      auto sharder_tensor_equal = [&](int64_t start, int64_t end) {
        SpecialCompute<T>(start, end, input0, input1, output);
      };

      if (max_core_num == 0) {
        KERNEL_LOG_ERROR("max_core_num could not be 0.");
      }
      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                      sharder_tensor_equal),
          "TensorEqual Compute failed.")
    } else {
      SpecialCompute<T>(0, data_num, input0, input1, output);
    }
  }
  else{
  *output = false;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTensorEqual, TensorEqualCpuKernel);
}  // namespace aicpu