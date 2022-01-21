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

#include "inv.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kInv = "Inv";
constexpr int64_t kParallelDataNums = 128 * 1024;

#define INV_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                     \
    uint32_t result = InvCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Inv kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }
}  // namespace

namespace aicpu {
uint32_t InvCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kInv);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    INV_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    INV_COMPUTE_CASE(DT_FLOAT, float, ctx)
    INV_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    INV_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    INV_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Inv kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InvCpuKernel::InvCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  size_t data_num = ctx.Input(0)->NumElements();    
  int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
  if(shape_x.size() == 1){
    if (data_size <= kParallelDataNums){
      for (size_t i = 0; i < data_num; i++) {
          T x = T(*(input_x + i));
          if (x == static_cast<T>(0)) {
            *(output_y + i) = static_cast<T>(INFINITY);
          } else {
            *(output_y + i) = static_cast<T>(1.0) / x;
          }
      }
    }
    else{
      uint32_t min_core_num = 1;
      uint32_t max_core_num =
          std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > data_num) {
        max_core_num = data_num;
      }
      auto shard_inv = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++){
          T x = T(*(input_x + i));
          if (x == static_cast<T>(0)) {
            *(output_y + i) = static_cast<T>(INFINITY);
          } else {
            *(output_y + i) = static_cast<T>(1.0) / x;
          }
        }
      };
      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                      shard_inv),
          "Inv Compute failed.")
    }   
  }
  else{
    int64_t m = shape_x[shape_size - 2];
    int64_t n = shape_x[shape_size - 1];
    using MartixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    if ( m * n > 0) {
      int64_t size_mn = m * n;    
      size_t martix_num = ctx.Input(0)->NumElements() / size_mn;

      if (data_size <= kParallelDataNums) {
        for (size_t i = 0; i < martix_num; i++) {
          Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
          Eigen::Map<MartixXd> martix_y(output_y + i * size_mn, m, n);
          martix_y = martix_x.cwiseInverse();
        }
      }      
      else {
        uint32_t min_core_num = 1;
        uint32_t max_core_num =
            std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        if (max_core_num > martix_num) {
          max_core_num = martix_num;
        }
        auto shard_inv = [&](size_t start, size_t end) {
          {
              for (size_t i = start; i < end; i++) {
                Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
                Eigen::Map<MartixXd> martix_y(output_y + i * size_mn, m, n);
                martix_y = martix_x.cwiseInverse();
              }
          }
        };
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, martix_num,
                                        martix_num / max_core_num, shard_inv),
            "Inv Compute failed.")
      }      
     }
    }
  return KERNEL_STATUS_OK;
 }  
REGISTER_CPU_KERNEL(kInv, InvCpuKernel);
}  // namespace aicpu