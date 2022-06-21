/**
 * Copyright(c) Huawei Technologies Co., Ltd.2021-2021.All rights reserved.
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

#include "triu.h"

#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "iostream"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kTriu = "Triu";
constexpr int64_t kParallelDataNums = 1024 * 1024;
const int32_t kTwo = 2;

#define TRIU_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = DoCompute<TYPE>(CTX);            \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Triu kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t TriuCpuKernel::ValidParam(CpuKernelContext &ctx) {
  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto output_shape = ctx.Output(0)->GetTensorShape();
  auto input_dims = input_shape->GetDims();

  KERNEL_CHECK_FALSE(input_dims >= kTwo, KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 2, but got rank [%d]",
                     input_shape->GetDims());

  auto input_data_type = ctx.Input(0)->GetDataType();
  auto output_data_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE(
      input_data_type == output_data_type, KERNEL_STATUS_PARAM_INVALID,
      "The data type of input [%s] need be same with output [%s].",
      DTypeStr(input_data_type).c_str(), DTypeStr(output_data_type).c_str())

  KERNEL_CHECK_FALSE(
      input_shape->GetDimSizes() == output_shape->GetDimSizes(),
      KERNEL_STATUS_PARAM_INVALID,
      "The output shape size should be same as the input shape size.");

  AttrValue *diagonal = ctx.GetAttr("diagonal");
  diagonal_ = (diagonal == nullptr) ? 0 : (diagonal->GetInt());
  KERNEL_LOG_DEBUG("%s Attr[diagonal] value[%d]", kTriu, diagonal_);

  return KERNEL_STATUS_OK;
}

template <typename MatrixMap>
void TriuCpuKernel::SetResultDiagonalMinus(MatrixMap output, MatrixMap input,
                                           int32_t diagonal_,
                                           int64_t matrix_height,
                                           int64_t matrix_width) {
  for (int j = 0; j < matrix_height; j++) {
    for (int i = j + 1; i <= j - diagonal_ && i < matrix_width; i++) {
      output(i, j) = input(i, j);
    }
  }
}

template <typename MatrixMap, typename T>
void TriuCpuKernel::SetResultDiagonaPositive(MatrixMap output,
                                             int32_t diagonal_,
                                             int64_t matrix_height,
                                             int64_t matrix_width) {
  for (int i = 0; i < matrix_width; i++) {
    for (int j = i; j < i + diagonal_ && j < matrix_height; j++) {
      output(i, j) = static_cast<T>(0);
    }
  }
}
template <typename T>
void TriuCpuKernel::SetResult(CpuKernelContext &ctx, int64_t matrix_start,
                              int64_t matrix_end) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);

  auto input_shape = input_tensor->GetTensorShape();

  auto input_dim_size = input_shape->GetDimSizes();

  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  auto input_dims = input_shape->GetDims();
  int64_t matrix_width = input_dim_size[input_dims - 2];
  int64_t matrix_height = input_dim_size[input_dims - 1];
  int64_t matrix_size = matrix_width * matrix_height;

  for (int64_t k = matrix_start; k < matrix_end; k++) {
    MatrixMap input(
        reinterpret_cast<T *>(input_tensor->GetData()) + k * matrix_size,
        matrix_width, matrix_height);
    MatrixMap output(
        reinterpret_cast<T *>(output_tensor->GetData()) + k * matrix_size,
        matrix_width, matrix_height);
    output = input.template triangularView<Eigen::Upper>();
    if (diagonal_ < 0) {
      SetResultDiagonalMinus<MatrixMap>(output, input, diagonal_, matrix_height,
                                        matrix_width);
    } else {
      SetResultDiagonaPositive<MatrixMap, T>(output, diagonal_, matrix_height,
                                             matrix_width);
    }
  }
}

template <typename T>
uint32_t TriuCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);

  auto input_shape = input_tensor->GetTensorShape();

  auto input_dim_size = input_shape->GetDimSizes();
  auto output_dim_size = input_shape->GetDimSizes();

  auto input_dims = input_shape->GetDims();
  int64_t matrix_width = input_dim_size[input_dims - 2];
  int64_t matrix_height = input_dim_size[input_dims - 1];
  int64_t matrix_size = matrix_width * matrix_height;
  int64_t matrixs_num = input_tensor->NumElements() / matrix_size;
  if (input_tensor->GetDataSize() <= kParallelDataNums) {
    SetResult<T>(ctx, 0, matrixs_num);
  } else {
    int64_t max_core_num = std::max(
        1, static_cast<int>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    if (max_core_num > matrixs_num) {
      max_core_num = matrixs_num;
    }
    auto shard_triu = [&](size_t start, size_t end) {
      SetResult<T>(ctx, start, end);
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(
        ctx, matrixs_num, matrixs_num / max_core_num, shard_triu);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t TriuCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Triu check input and output number failed.");
  KERNEL_HANDLE_ERROR(ValidParam(ctx), "[%s] check params failed.", kTriu);

  auto data_type = ctx.Input(0)->GetDataType();

  switch (data_type) {
    TRIU_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    TRIU_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    TRIU_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TRIU_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    TRIU_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    TRIU_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    TRIU_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    TRIU_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    TRIU_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    TRIU_COMPUTE_CASE(DT_FLOAT, float, ctx)
    TRIU_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    TRIU_COMPUTE_CASE(DT_BOOL, bool, ctx)
    default:
      KERNEL_LOG_ERROR("Triu kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTriu, TriuCpuKernel);
}  // namespace aicpu
