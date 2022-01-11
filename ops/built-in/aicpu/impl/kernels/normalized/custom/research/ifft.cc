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
#include "ifft.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kIFFT = "IFFT";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
constexpr int64_t kParallelDataNums = 4 * 1024;
constexpr int64_t kParallelDataNumsMid = 16 * 1024;

#define IFFT_COMPUTE_CASE(DTYPE, TYPE, type, CTX)      \
  case (DTYPE): {                                      \
    uint32_t result = IFFTCompute<TYPE, type>(CTX);    \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("IFFT kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t IFFTCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kIFFT);
  KERNEL_HANDLE_ERROR(IFFTCheck(ctx), "[%s] check params failed.", kIFFT);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    IFFT_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, float, ctx)
    IFFT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, double, ctx)
    default:
      KERNEL_LOG_ERROR("IFFT kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t IFFTCpuKernel::IFFTCheck(CpuKernelContext &ctx) {
  auto input = ctx.Input(0);
  auto output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed")
  KERNEL_CHECK_NULLPTR(input->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input tensor shape failed.")
  std::vector<int64_t> shape_in = input->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_in.size() >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_in.size())
  return KERNEL_STATUS_OK;
}

template <typename T, typename t>
uint32_t IFFTCpuKernel::IFFTCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> shape_in = ctx.Input(0)->GetTensorShape()->GetDimSizes();

  size_t shape_size = shape_in.size();
  int64_t m = shape_in.at(shape_size - 1);
  if (m > 0) {
    size_t martix_num = ctx.Input(0)->NumElements() / m;
    int64_t data_size = ctx.Input(0)->NumElements();
    if (data_size <= kParallelDataNums) {
      std::vector<T> tmpIn;
      std::vector<T> tmpOut;
      Eigen::FFT<t, Eigen::default_fft_impl<t>> fft;
      for (size_t i = 0; i < martix_num; i++) {
        for (int64_t j = 0; j < m; j++) {
          tmpIn.push_back(*(input_x + i * m + j));
        }
        fft.inv(tmpOut, tmpIn, -1);
        for (int64_t j = 0; j < m; j++) {
          *(output_y + i * m + j) = tmpOut[j];
        }
        tmpIn.clear();
        tmpOut.clear();
      }
    } else {
      uint32_t min_core_num = 1;
      uint32_t max_core_num =
          std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (data_size <= kParallelDataNumsMid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }
      if (max_core_num > martix_num) {
        max_core_num = martix_num;
      }
      auto shard_ifft = [&](size_t start, size_t end) {
        std::vector<T> tmpIn;
        std::vector<T> tmpOut;
        Eigen::FFT<t, Eigen::default_fft_impl<t>> fft;
        for (size_t i = start; i < end; i++) {
          for (int64_t j = 0; j < m; j++) {
            tmpIn.push_back(*(input_x + i * m + j));
          }
          fft.inv(tmpOut, tmpIn, -1);
          for (int64_t j = 0; j < m; j++) {
            *(output_y + i * m + j) = tmpOut[j];
          }
          tmpIn.clear();
          tmpOut.clear();
        }
      };
      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, martix_num,
                                      martix_num / max_core_num, shard_ifft),
          "IFFT Compute failed.")
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kIFFT, IFFTCpuKernel);
}  // namespace aicpu