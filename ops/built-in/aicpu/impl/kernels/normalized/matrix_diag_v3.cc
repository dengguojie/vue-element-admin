/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "matrix_diag_v3.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>

#include "log.h"
#include "utils/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
const int64_t kConstTwo = 2;
const char *const MATRIX_DIAG_V3 = "MatrixDiagV3";

#define MATRIXDIAGV3_COMPUTE_CASE(DTYPE, TYPE, CTX)             \
  case (DTYPE): {                                               \
    uint32_t result = DoCompute<TYPE>(CTX);                     \
    if (result != KERNEL_STATUS_OK) {                           \
      KERNEL_LOG_ERROR("MatrixDiagV3 kernel compute failed.");  \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}

namespace aicpu {
uint32_t MatrixDiagV3CpuKernel::CheckParam(const CpuKernelContext &ctx) {
  std::string align = "RIGHT_LEFT";
  AttrValue *attr_align = ctx.GetAttr("align");
  if (attr_align != NULL) {
    align = attr_align->GetString();
  }

  KERNEL_CHECK_FALSE(
      (align == "" || align == "RIGHT_LEFT" || align == "RIGHT_RIGHT" ||
      align == "LEFT_LEFT" || align == "LEFT_RIGHT"),
      KERNEL_STATUS_PARAM_INVALID,
      "Attr 'align' of 'MatrixDiagV3' is not in: 'LEFT_RIGHT', "
      "'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'.");
  left_align_superdiagonal = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
  left_align_subdiagonal = align == "LEFT_LEFT" || align == "RIGHT_LEFT";

  auto diagonal_data_type = ctx.Input(0)->GetDataType();
  auto output_data_type = ctx.Output(0)->GetDataType();

  KERNEL_CHECK_FALSE(
      diagonal_data_type == output_data_type,
      KERNEL_STATUS_PARAM_INVALID,
      "The data type of input0 [%s] need be same with output0 [%s].",
      DTypeStr(diagonal_data_type).c_str(), DTypeStr(output_data_type).c_str());

  return KERNEL_STATUS_OK;
}

uint32_t MatrixDiagV3CpuKernel::Compute(CpuKernelContext &ctx)
{
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
      "MatrixDiagV3 check input and output number failed.");
  KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK),
      KERNEL_STATUS_PARAM_INVALID, "CheckParam failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MATRIXDIAGV3_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_FLOAT, std::float_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_DOUBLE, std::double_t, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_COMPLEX128, std::complex<std::double_t>, ctx)
    MATRIXDIAGV3_COMPUTE_CASE(DT_COMPLEX64, std::complex<std::float_t>, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixDiagV3 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

std::pair<int, int> MatrixDiagV3CpuKernel::ComputeDiagLenAndContentOffset(
    int diag_index, int max_diag_len, int num_rows, int num_cols,
    bool left_aligns_superdiagonal, bool left_aligns_subdiagonal) const {
  const bool left_align = (diag_index >= 0 && left_aligns_superdiagonal) ||
                          (diag_index <= 0 && left_aligns_subdiagonal);
  const int diag_len = std::min(num_rows + std::min(0, diag_index),
                                num_cols - std::max(0, diag_index));
  const int content_offset = (left_align) ? 0 : (max_diag_len - diag_len);
  return { diag_len, content_offset };
}

uint32_t MatrixDiagV3CpuKernel::GetDiagIndex(const CpuKernelContext &ctx,
                                             int32_t &lower_diag_index,
                                             int32_t &upper_diag_index,
                                             int32_t &num_rows,
                                             int32_t &num_cols) const {
  Tensor *num_rows_tensor = ctx.Input(2);
  Tensor *num_cols_tensor = ctx.Input(3);
  Tensor *k_tensor = ctx.Input(1);
  auto *k_data = reinterpret_cast<int32_t *>(k_tensor->GetData());
  lower_diag_index = k_data[0];
  upper_diag_index = lower_diag_index;
  int64_t k_num = k_tensor->NumElements();
  if (k_num <= 0 || k_num > kConstTwo) {
    KERNEL_LOG_ERROR("k must have only one or two elements, received ",
                     "[%d] elements.", k_num);
    return KERNEL_STATUS_PARAM_INVALID;
  } else if (k_num == kConstTwo) {
    upper_diag_index = k_data[1];
  }
  KERNEL_CHECK_FALSE((lower_diag_index <= upper_diag_index), KERNEL_STATUS_PARAM_INVALID,
      "lower_diag_index must be smaller than upper_diag_index,received [%d] is larger than [%d] ",
      lower_diag_index, upper_diag_index);

  // num_rows
  int64_t num_rows_num = num_rows_tensor->NumElements();
  KERNEL_CHECK_FALSE((num_rows_num == 1), KERNEL_STATUS_PARAM_INVALID,
      "num_rows must have only one element, received [%d] elements. ", num_rows_num);
  auto *num_rows_data = reinterpret_cast<int32_t *>(num_rows_tensor->GetData());
  num_rows = num_rows_data[0];

  // num_cols
  int64_t num_cols_num = num_cols_tensor->NumElements();
  KERNEL_CHECK_FALSE((num_cols_num == 1), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
      "num_cols must have only one element, received [%d] elements. ", num_cols_num);
  auto *num_cols_data = reinterpret_cast<int32_t *>(num_cols_tensor->GetData());
  num_cols = num_cols_data[0];
  return KERNEL_STATUS_OK;
}

uint32_t MatrixDiagV3CpuKernel::AdjustRowsAndCols(int32_t &num_rows,
                                                  int32_t &num_cols,
                                                  int32_t min_num_rows,
                                                  int32_t min_num_cols) const {
  if (num_rows == -1 && num_cols == -1) {
    num_rows = std::max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  } else if (num_rows == -1) {
    num_rows = min_num_rows;
  } else if (num_cols == -1) {
    num_cols = min_num_cols;
  }
  if (num_rows != min_num_rows && num_cols != min_num_cols) {
    KERNEL_LOG_ERROR("The number of rows or columns is not consistent with "
                     "the specified d_lower, d_upper, and diagonal.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDiagV3CpuKernel::DoCompute(const CpuKernelContext &ctx) {
  Tensor *diagonal_tensor = ctx.Input(0);
  Tensor *padding_value_tensor = ctx.Input(4);
  Tensor *output_tensor = ctx.Output(0);
  auto diagonal_shape = diagonal_tensor->GetTensorShape();
  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  int32_t num_rows = -1;
  int32_t num_cols = -1;
  
  KERNEL_CHECK_FALSE((GetDiagIndex(ctx, lower_diag_index, upper_diag_index, num_rows, num_cols) == KERNEL_STATUS_OK),
      static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "GetDiagIndex failed.");

  // padding_value
  int64_t padding_value_num = padding_value_tensor->NumElements();
  KERNEL_CHECK_FALSE((padding_value_num == 1), KERNEL_STATUS_PARAM_INVALID,
      "padding_value must have only one element, received [%d] elements. ", padding_value_num);
  auto *padding_value_data = reinterpret_cast<T *>(padding_value_tensor->GetData());
  T padding_value = padding_value_data[0];

  const int32_t diag_rank = diagonal_shape->GetDims();
  const int64_t num_diags = (upper_diag_index - lower_diag_index) + 1;
  const int64_t max_diag_len = diagonal_shape->GetDimSize(diag_rank - 1);
  const int32_t min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
  const int32_t min_num_cols = max_diag_len + std::max(lower_diag_index, 0);
  if (num_rows != -1 && num_rows < min_num_rows) {
    KERNEL_LOG_ERROR("The number of rows is too small.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (num_cols != -1 && num_cols < min_num_cols) {
    KERNEL_LOG_ERROR("The number of columns is too small.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  // If both num_rows and num_cols are unknown, assume that output is square.
  // Otherwise, use smallest possible values.
  KERNEL_CHECK_FALSE((AdjustRowsAndCols(num_rows, num_cols, min_num_rows, min_num_cols) == KERNEL_STATUS_OK),
      static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "AdjustRowsAndCols failed.");

  auto *diagonal_data = reinterpret_cast<T *>(diagonal_tensor->GetData());
  auto *output_data = reinterpret_cast<T *>(output_tensor->GetData());
  int64_t diag_elements_in_batch = num_diags * max_diag_len;
  int64_t diag_batch_base_index = 0 * diag_elements_in_batch;
  uint64_t num_element = output_tensor->NumElements();
  uint64_t num_batches = num_element / (num_rows * num_cols);
  int64_t elem = 0;

  for (uint64_t batch = 0; batch < num_batches ; ++batch) {
    for (int64_t i = 0; i < num_rows; ++i) {
      for (int64_t j = 0; j < num_cols; ++j) {
        const int diag_index = static_cast<int>(j - i);
        const int diag_index_in_input = upper_diag_index - diag_index;
        int diag_len, content_offset;
        std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
            diag_index, static_cast<int>(max_diag_len), num_rows, num_cols,
            left_align_superdiagonal, left_align_subdiagonal);
        const int index_in_the_diagonal =
          (j - std::max<int64_t>(diag_index, 0)) + content_offset;
        if (lower_diag_index <= diag_index &&
            diag_index <= upper_diag_index) {
          output_data[elem] = diagonal_data[diag_batch_base_index +
            diag_index_in_input * max_diag_len +
            index_in_the_diagonal];
          elem++;
        } else {
          output_data[elem] = padding_value;
          elem++;
        }
      }
    }
    diag_batch_base_index += diag_elements_in_batch;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(MATRIX_DIAG_V3, MatrixDiagV3CpuKernel);
} // namespace aicpu

