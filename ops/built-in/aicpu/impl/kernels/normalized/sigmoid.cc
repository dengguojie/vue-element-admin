/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All right reserved.
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
#include "sigmoid.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cmath"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kSigmoid = "Sigmoid";
constexpr int64_t kParallelDataNums = 16 * 1024;

#define SIGMOID_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = SigmoidCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Sigmoid kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }

#define SIGMOID_COMPUTE_CASE2(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                         \
    uint32_t result = SigmoidComputeComplex<TYPE>(CTX);   \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Sigmoid kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t SigmoidCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.",kSigmoid);
  KERNEL_HANDLE_ERROR(SigmoidCheck(ctx),
                      "[%s] check params failed.", kSigmoid);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SIGMOID_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SIGMOID_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SIGMOID_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SIGMOID_COMPUTE_CASE2(DT_COMPLEX64, std::complex<float>, ctx)
    SIGMOID_COMPUTE_CASE2(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Sigmoid kernel data type [%s] not support.",
                      DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SigmoidCpuKernel::SigmoidCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                      "Get input data failed.")
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                      "Get output data failed")
  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                      "Get input tensor shape failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SigmoidCpuKernel::SigmoidCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= kParallelDataNums){
    for (int64_t i = 0; i < data_num; i++) {
      *(output_y + i) = static_cast<T>(1) / 
                        (static_cast<T>(1) + (static_cast<T>(1) / 
                        exp(*(input_x + i))));
    }
  }else{
    uint32_t min_core_num = 1;
    int64_t max_core_num =
    	std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_sigmoid = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(output_y + i) = static_cast<T>(1) / 
                          (static_cast<T>(1) + (static_cast<T>(1) / 
                          exp(*(input_x + i))));;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sigmoid),
                        "Sigmoid Compute failed.")
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SigmoidCpuKernel::SigmoidComputeComplex(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>   ArrayxXd;
  ArrayxXd array_x(1, data_num);

  if(data_size <= kParallelDataNums){
    for (int64_t i = 0; i < data_num; i++){
      *(output_y + i) = static_cast<T>(1) / 
                        (static_cast<T>(1) + (static_cast<T>(1) / 
                        Eigen::numext::exp(*(input_x + i))));
    }
    return KERNEL_STATUS_OK;
  }else{
    uint32_t min_core_num = 1;
    int64_t max_core_num =
            std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
        max_core_num = data_num;
    }
    auto shard_sigmoid = [&](size_t start, size_t end){
      for (size_t i = start; i < end; i++){
        *(output_y + i) = static_cast<T>(1) / 
                          (static_cast<T>(1) + (static_cast<T>(1) / 
                          Eigen::numext::exp(*(input_x + i))));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sigmoid),
                        "Sigmoid Compute failed.")
    return KERNEL_STATUS_OK;
    }
}
REGISTER_CPU_KERNEL(kSigmoid, SigmoidCpuKernel);
}  // namespace aicpu