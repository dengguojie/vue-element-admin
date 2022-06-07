/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "matrix_diag_part_v3.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *MATRIX_DIAG_PART_V3 = "MatrixDiagPartV3";
const int64_t zero = 0;
constexpr int64_t kDimSize = 2;
constexpr int64_t InputDimSize = 2;
constexpr int64_t kInputNum = 3;
constexpr int64_t kOutputNum = 1;
constexpr int64_t kTwoNum = 2;
static std::pair<int64_t, int64_t> ComputeDiagLenAndContentOffset(
    int64_t diag_index, int64_t max_diag_len, int64_t num_rows,
    int64_t num_cols, bool left_align_superdiagonal,
    bool left_align_subdiagonal) {
  int64_t zero = 0;
  const bool left_align = (diag_index >= 0 && left_align_superdiagonal) ||
                          (diag_index <= 0 && left_align_subdiagonal);
  const int diag_len = std::min(num_rows + std::min(zero, diag_index),
                                num_cols - std::max(zero, diag_index));
  const int64_t content_offset = (left_align) ? 0 : (max_diag_len - diag_len);
  return {diag_len, content_offset};
}
}  // namespace

namespace aicpu {
uint32_t MatrixDiagPartV3CpuKernel::CheckParam(CpuKernelContext &ctx) {
  auto output_data_temp = ctx.Output(0)->GetData();
  Tensor *input_tensor = ctx.Input(0);
  Tensor *k = ctx.Input(1);
  Tensor *padding_value = ctx.Input(2);
  auto padding_value_data = padding_value->GetData();
  // check output
  KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output data failed.", MATRIX_DIAG_PART_V3);
  // check padding_value
  KERNEL_CHECK_NULLPTR(padding_value_data, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get paddingvalue data failed.",
                       MATRIX_DIAG_PART_V3);
  // check k
  auto k_shape = k->GetTensorShape();
  std::vector<int64_t> k_dims = k_shape->GetDimSizes();
  KERNEL_CHECK_NULLPTR(k, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get k data failed.", MATRIX_DIAG_PART_V3);
  KERNEL_CHECK_FALSE((k_dims.size() <= kDimSize), KERNEL_STATUS_PARAM_INVALID,
                     "%s dims size [%zu] must <= 2.", MATRIX_DIAG_PART_V3,
                     k_dims.size());
  // check align
  std::string align = "RIGHT_LEFT";
  AttrValue *attr_align = ctx.GetAttr("align");
  if (attr_align != nullptr) {
    align = attr_align->GetString();
  }
  if (align == "LEFT_LEFT" || align == "LEFT_RIGHT") {
    left_align_superdiagonal = true;
  } else {
    left_align_superdiagonal = false;
  }
  if (align == "LEFT_LEFT" || align == "RIGHT_LEFT") {
    left_align_subdiagonal = true;
  } else {
    left_align_subdiagonal = false;
  }
  KERNEL_CHECK_FALSE(
      (align == "RIGHT_LEFT" || align == "LEFT_LEFT" ||
       align == "RIGHT_RIGHT" || align == "LEFT_RIGHT"),
      KERNEL_STATUS_PARAM_INVALID,
      "align must be one of RIGHT_LEFT,LEFT_LEFT,RIGHT_RIGHT,LEFT_RIGHT.");
  // check input
  auto input_shape = input_tensor->GetTensorShape();
  std::vector<int64_t> input_dims = input_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(
      (input_dims.size() >= InputDimSize), KERNEL_STATUS_PARAM_INVALID,
      "input dims must >=2 while %d", input_dims.size());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::MultiProcessFunc(CpuKernelContext &ctx,
                                                     int64_t upper_diag_index,
                                                     int64_t num_diags,
                                                     int64_t max_diag_len,
                                                     int64_t num_rows,
                                                     int64_t num_cols,
                                                     int64_t num_array) {
  auto inputData = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto outputData = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto paddingvalue = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  const int64_t output_elements_in_batch = num_diags * max_diag_len;
  uint32_t min_core_num = 1;
  uint32_t max_core_num =
      std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
  if (max_core_num > num_array) {
    max_core_num = num_array;
  }
  auto shard = [&](size_t start, size_t end) {
    int64_t output_base_index = start * output_elements_in_batch;
    for (size_t batch = start; batch < end; ++batch) {
      for (int64_t m = 0; m < num_diags; ++m) {
        int64_t content_offset = 0;
        int64_t diag_len = 0;
        int64_t diag_index = upper_diag_index - m;
        int64_t y_offset = std::max(zero, -diag_index);
        int64_t x_offset = std::max(zero, diag_index);
        std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
            diag_index, max_diag_len, num_rows, num_cols,
            left_align_superdiagonal, left_align_subdiagonal);
        // Fills the diagonal.
        for (int64_t n = 0; n < diag_len; n++) {
          outputData[output_base_index + content_offset + n] =
              inputData[batch * num_rows * num_cols +
                        (n + y_offset) * num_cols + n + x_offset];
        }
        // Padding.
        const bool left_align = (content_offset == 0);
        const int64_t padding_start = (left_align) ? diag_len : 0;
        const int64_t padding_end =
            (left_align) ? max_diag_len : content_offset;
        int64_t n = padding_start;
        while (n < padding_end) {
          outputData[output_base_index + n] = paddingvalue[0];
          n += 1;
        }
        output_base_index += max_diag_len;
      }
    }
  };
  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0.");
  }
  CpuKernelUtils::ParallelFor(ctx, num_array, num_array / max_core_num,
                              shard);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::SingleProcessFunc(CpuKernelContext &ctx,
                                                      int64_t upper_diag_index,
                                                      int64_t num_diags,
                                                      int64_t max_diag_len,
                                                      int64_t num_rows,
                                                      int64_t num_cols,
                                                      int64_t num_array) {
  auto inputData = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto outputData = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto paddingvalue = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  int64_t output_base_index = 0;
  for (int64_t batch = 0; batch < num_array; ++batch) {
    for (int64_t m = 0; m < num_diags; ++m) {
      int64_t content_offset = 0;
      int64_t diag_len = 0;
      int64_t diag_index = upper_diag_index - m;
      int64_t y_offset = std::max(zero, -diag_index);
      int64_t x_offset = std::max(zero, diag_index);
      std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
          diag_index, max_diag_len, num_rows, num_cols,
          left_align_superdiagonal, left_align_subdiagonal);
      // Fills the diagonal.
      for (int64_t n = 0; n < diag_len; n++) {
        outputData[output_base_index + content_offset + n] =
            inputData[batch * num_rows * num_cols +
                      (n + y_offset) * num_cols + n + x_offset];
      }
      // Padding.
      const bool left_align = (content_offset == 0);
      const int64_t padding_start = (left_align) ? diag_len : 0;
      const int64_t padding_end =
          (left_align) ? max_diag_len : content_offset;
      int64_t n = padding_start;
      while (n < padding_end) {
        outputData[output_base_index + n] = paddingvalue[0];
        n += 1;
      }
      output_base_index += max_diag_len;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *k = ctx.Input(1);
  auto k_Data = reinterpret_cast<int32_t *>(k->GetData());

  int64_t len_k = k->NumElements();
  int64_t lower_diag_index = 0;
  int64_t upper_diag_index = 0;
  lower_diag_index = k_Data[0];
  upper_diag_index = k_Data[0];

  if (len_k == kTwoNum) {
    upper_diag_index = k_Data[1];
  }
  KERNEL_CHECK_FALSE((lower_diag_index <= upper_diag_index),
                     KERNEL_STATUS_PARAM_INVALID,
                     " k[0] must not be larger than k[1] .");
  auto input_shape = input->GetTensorShape();
  int64_t rank = input_shape->GetDims();
  if (rank < kTwoNum) {
    KERNEL_LOG_ERROR("input dims must >=2");
  }
  int64_t num_rows = input_shape->GetDimSize(rank - 2);
  int64_t num_cols = input_shape->GetDimSize(rank - 1);
  int64_t num_array = input_shape->NumElements() / (num_rows * num_cols);
  int64_t num_diags = upper_diag_index - lower_diag_index + 1;
  int64_t max_diag_len = std::min(num_rows + std::min(upper_diag_index, zero),
                                  num_cols - std::max(lower_diag_index, zero));
  const int64_t output_elements_in_batch = num_diags * max_diag_len;
  const int64_t data_num = num_array * output_elements_in_batch;
  const int64_t kParallelArrayNumSameShape = 2048;
  if (data_num >= kParallelArrayNumSameShape) {
    return MultiProcessFunc<T>(ctx, upper_diag_index, num_diags,
                          max_diag_len, num_rows, num_cols, num_array);
  } else {
    return SingleProcessFunc<T>(ctx, upper_diag_index, num_diags,
                           max_diag_len, num_rows, num_cols, num_array);
  }
}

uint32_t MatrixDiagPartV3CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "MatrixDiagPartV3 check input and output number failed.");
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckParam failed.");
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  switch (dt) {
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_FLOAT:
      return DoCompute<std::float_t>(ctx);
    case DT_DOUBLE:
      return DoCompute<std::double_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<std::double_t>>(ctx);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    default:
      KERNEL_LOG_ERROR(
          "MatrixDiagPartV3 kernels does not support this data type [%s].",
          DTypeStr(dt).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
REGISTER_CPU_KERNEL(MATRIX_DIAG_PART_V3, MatrixDiagPartV3CpuKernel);
}  // namespace aicpu
