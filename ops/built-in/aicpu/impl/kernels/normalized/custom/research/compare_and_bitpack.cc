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

#include "compare_and_bitpack.h"

#include <vector>

#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kCompareAndBitpack = "CompareAndBitpack";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
constexpr int64_t kParallelDataNums = 64 * 1024;

#define COMPARE_COMPUTE_CASE(DTYPE, TYPE, X, Y, Z, CTX)             \
  case (DTYPE): {                                                   \
    uint32_t result = CompareCompute<TYPE>(X, Y, Z, CTX);           \
    if (result != KERNEL_STATUS_OK) {                               \
      KERNEL_LOG_ERROR("CompareAndBitpack kernel compute failed."); \
      return result;                                                \
    }                                                               \
    break;                                                          \
  }
}  // namespace

namespace aicpu {
uint32_t CompareAndBitpackCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(
      NormalCheck(ctx, kInputNum, kOutputNum),
      "CompareAndBitpack check input and output number failed.");
  Tensor *x = ctx.Input(0);
  Tensor *threshold = ctx.Input(1);
  Tensor *y = ctx.Output(0);
  uint32_t ret = CheckParam(x, threshold, y, ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  return KERNEL_STATUS_OK;
}

uint32_t CompareAndBitpackCpuKernel::CheckParam(Tensor *x, Tensor *y, Tensor *z,
                                                CpuKernelContext &ctx) {
  auto input_shape = x->GetTensorShape();
  auto rank = input_shape->GetDims();
  if (input_shape->GetDimSizes().empty()) {
    KERNEL_LOG_ERROR("Input tensor should not be a Scalar.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto inest_dim = input_shape->GetDimSize(rank - 1);
  if ((inest_dim % 8) != 0) {
    KERNEL_LOG_ERROR(
        "The innermost dimension of input "
        "tensor must be divisible by 8.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (!y->GetTensorShape()->GetDimSizes().empty()) {
    KERNEL_LOG_ERROR("Input threshold must be a Scalar.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto output_shape = z->GetTensorShape();
  if (output_shape->GetDims() != rank ||
      output_shape->GetDimSize(rank - 1) != inest_dim % 8) {
    std::vector<int64_t> output_dim = input_shape->GetDimSizes();
    output_dim[rank - 1] /= 8;
    output_shape->SetDimSizes(output_dim);
    z->SetTensorShape(output_shape.get());
  }
  DataType input_type = x->GetDataType();
  DataType thresh_type = y->GetDataType();
  KERNEL_CHECK_FALSE(
      (input_type == thresh_type), KERNEL_STATUS_PARAM_INVALID,
      "The data type of input [%s] need be same with threshold [%s].",
      DTypeStr(input_type).c_str(), DTypeStr(thresh_type).c_str())
  switch (input_type) {
    COMPARE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_FLOAT, float, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_DOUBLE, double, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_INT8, int8_t, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_INT16, int16_t, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_INT32, int32_t, x, y, z, ctx)
    COMPARE_COMPUTE_CASE(DT_INT64, int64_t, x, y, z, ctx)
    case DT_BOOL:
      return BoolCompute(x, y, z, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("CompareAndBitpack kernel data type [%s] not support.",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CompareAndBitpackCpuKernel::CompareCompute(Tensor *x, Tensor *y,
                                                    Tensor *z,
                                                    CpuKernelContext &ctx) {
  T *input_addr = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(input_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.");
  T *threshold_addr = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(threshold_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get threshold data failed.");
  uint8_t *output_addr = reinterpret_cast<uint8_t *>(z->GetData());
  KERNEL_CHECK_NULLPTR(output_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed");

  uint64_t data_num = x->GetDataSize() / sizeof(T) / 8;
  T thresh = *(threshold_addr);
  if (data_num < kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      const T *x_index = input_addr + i * 8;
      *(output_addr + i) =
          (((*(x_index) > thresh) << 7) | ((*(x_index + 1) > thresh) << 6) |
           ((*(x_index + 2) > thresh) << 5) | ((*(x_index + 3) > thresh) << 4) |
           ((*(x_index + 4) > thresh) << 3) | ((*(x_index + 5) > thresh) << 2) |
           ((*(x_index + 6) > thresh) << 1) | ((*(x_index + 7) > thresh)));
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_compare = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        const T *x_index = input_addr + i * 8;
        *(output_addr + i) =
            (((*(x_index) > thresh) << 7) | ((*(x_index + 1) > thresh) << 6) |
             ((*(x_index + 2) > thresh) << 5) |
             ((*(x_index + 3) > thresh) << 4) |
             ((*(x_index + 4) > thresh) << 3) |
             ((*(x_index + 5) > thresh) << 2) |
             ((*(x_index + 6) > thresh) << 1) | ((*(x_index + 7) > thresh)));
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    shard_compare),
        "CompareAndBitpack Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

uint32_t CompareAndBitpackCpuKernel::BoolCompute(Tensor *x, Tensor *y,
                                                 Tensor *z,
                                                 CpuKernelContext &ctx) {
  if (sizeof(bool) == 1) {
    bool *input_addr = reinterpret_cast<bool *>(x->GetData());
    KERNEL_CHECK_NULLPTR(input_addr, KERNEL_STATUS_PARAM_INVALID,
                         "Get input data failed.");
    bool *threshold_addr = reinterpret_cast<bool *>(y->GetData());
    KERNEL_CHECK_NULLPTR(threshold_addr, KERNEL_STATUS_PARAM_INVALID,
                         "Get threshold data failed.");
    uint8_t *output_addr = reinterpret_cast<uint8_t *>(z->GetData());
    KERNEL_CHECK_NULLPTR(output_addr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output data failed");

    uint64_t data_num = x->GetDataSize() / 8;
    if (data_num < kParallelDataNums) {
      for (size_t i = 0; i < data_num; i++) {
        const bool *x_index = input_addr + i * 8;
        *(output_addr + i) =
            (((*(x_index)&1) << 7) | ((*(x_index + 1) & 1) << 6) |
             ((*(x_index + 2) & 1) << 5) | ((*(x_index + 3) & 1) << 4) |
             ((*(x_index + 4) & 1) << 3) | ((*(x_index + 5) & 1) << 2) |
             ((*(x_index + 6) & 1) << 1) | ((*(x_index + 7) & 1)));
      }
    } else {
      uint32_t min_core_num = 1;
      uint64_t max_core_num =
          std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
      if (max_core_num > data_num) {
        max_core_num = data_num;
      }
      auto shard_bool = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          const bool *x_index = input_addr + i * 8;
          *(output_addr + i) =
              (((*(x_index)&1) << 7) | ((*(x_index + 1) & 1) << 6) |
               ((*(x_index + 2) & 1) << 5) | ((*(x_index + 3) & 1) << 4) |
               ((*(x_index + 4) & 1) << 3) | ((*(x_index + 5) & 1) << 2) |
               ((*(x_index + 6) & 1) << 1) | ((*(x_index + 7) & 1)));
        }
      };
      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                      shard_bool),
          "CompareAndBitpack Compute failed.");
    }

    return KERNEL_STATUS_OK;
  } else {
    return CompareCompute<bool>(x, y, z, ctx);
  }
}

REGISTER_CPU_KERNEL(kCompareAndBitpack, CompareAndBitpackCpuKernel);
}  // namespace aicpu
