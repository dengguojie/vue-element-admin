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

#include "matrix_set_diag_v2.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMatrixSetDiagV2 = "MatrixSetDiagV2";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;

#define SET_DIAG_CASE(DTYPE, TYPE, X, DIAG, K, Y, SIZE, NUM, LOWER, UPPER,  \
                      ROW, COL, MAXDIAG, CTX)                               \
  case (DTYPE): {                                                           \
    uint32_t result = SetDiagCompute<TYPE>(X, DIAG, K, Y, SIZE, NUM, LOWER, \
                                           UPPER, ROW, COL, MAXDIAG, CTX);  \
    if (result != KERNEL_STATUS_OK) {                                       \
      KERNEL_LOG_ERROR("MatrixSetDiagV2 kernel compute failed.");           \
      return result;                                                        \
    }                                                                       \
    break;                                                                  \
  }

#define CHECK_DIAG_SHAPE(X, DIAG, RANK, INESTRANK, MAXDIAG)                   \
  for (int i = 0; i < (RANK); i++) {                                          \
    if ((X)[i] != (DIAG)[i]) {                                                \
      KERNEL_LOG_ERROR(                                                       \
          "The %d-th dim of diagnoal must be %d, but get: "                   \
          "%d.",                                                              \
          i + 1, (X)[i], (DIAG)[i]);                                          \
      return KERNEL_STATUS_PARAM_INVALID;                                     \
    }                                                                         \
  }                                                                           \
  KERNEL_CHECK_FALSE(((DIAG)[INESTRANK] == (MAXDIAG)),                        \
                     KERNEL_STATUS_PARAM_INVALID,                             \
                     "The innermost dim of diagonal must be %d, but get %d.", \
                     MAXDIAG, (DIAG)[INESTRANK]);
}  // namespace

namespace aicpu {
uint32_t MatrixSetDiagV2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "MatrixSetDiagV2 check input and output number failed.");
  uint32_t ret = CheckParam(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  return KERNEL_STATUS_OK;
}

uint32_t MatrixSetDiagV2CpuKernel::CheckParam(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *diag = ctx.Input(1);
  Tensor *k = ctx.Input(2);
  Tensor *y = ctx.Output(0);

  int32_t rank = x->GetTensorShape()->GetDims();
  int32_t diag_rank = diag->GetTensorShape()->GetDims();
  std::vector<int64_t> input_dim = x->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> diag_dim = diag->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((rank >= 2), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least 2-dim, but get dims: %d", rank);
  int32_t k_size = k->GetDataSize() / sizeof(int32_t);
  int32_t *k_addr = reinterpret_cast<int32_t *>(k->GetData());
  KERNEL_CHECK_NULLPTR(k_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input k data failed.");
  int32_t num_row = input_dim[rank - 2];
  int32_t num_col = input_dim[rank - 1];
  int32_t num_diags = 1;
  int32_t max_diag_len = std::min(num_row + std::min(k_addr[0], 0),
                                  num_col - std::max(k_addr[0], 0));
  int32_t lower_diag = k_addr[0];
  int32_t upper_diag = lower_diag;
  if (k_size == 1) {
    KERNEL_CHECK_FALSE((rank - 1 == diag_rank), KERNEL_STATUS_PARAM_INVALID,
                       "Diagonal must be r-dim when k is scalar, input is "
                       "r+1-dim, but get dims: %d",
                       diag_rank);
    CHECK_DIAG_SHAPE(input_dim, diag_dim, rank - 2, rank - 2, max_diag_len);
  } else if (k_size == 2) {
    if (k_addr[0] == k_addr[1]) {
      KERNEL_CHECK_FALSE((rank - 1 == diag_rank), KERNEL_STATUS_PARAM_INVALID,
                         "Diagonal must be r-dim when k[0] == k[1], input is "
                         "r+1-dim, but get dims: %d",
                         diag_rank);
      CHECK_DIAG_SHAPE(input_dim, diag_dim, rank - 2, rank - 2, max_diag_len);
    } else if (k_addr[0] < k_addr[1]) {
      KERNEL_CHECK_FALSE((rank == diag_rank), KERNEL_STATUS_PARAM_INVALID,
                         "Diagonal must be r+1-dim when k[0] != k[1], input is "
                         "r+1-dim, but get dims: %d",
                         diag_rank);
      num_diags = k_addr[1] - k_addr[0] + 1;
      KERNEL_CHECK_FALSE((diag_dim[diag_rank - 2] == num_diags),
                         KERNEL_STATUS_PARAM_INVALID,
                         "The %d-th dim of diagonal must be %d, but get %d.",
                         diag_rank - 1, num_diags, diag_dim[diag_rank - 2]);
      max_diag_len = std::min(num_row + std::min(k_addr[1], 0),
                              num_col - std::max(k_addr[0], 0));
      CHECK_DIAG_SHAPE(input_dim, diag_dim, rank - 2, rank - 1, max_diag_len);
      upper_diag = k_addr[1];
    } else {
      KERNEL_LOG_ERROR("`k[0]` must not be larger than `k[1]`.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    KERNEL_LOG_ERROR("k must have 1 or 2 data, but get: %d data.", k_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint64_t x_size = x->GetDataSize();
  uint64_t y_size = y->GetDataSize();
  KERNEL_CHECK_FALSE((x_size == y_size), KERNEL_STATUS_PARAM_INVALID,
                     "Input data size[%llu] is not equal to output data "
                     "size[%llu].",
                     x_size, y_size);
  DataType x_type = x->GetDataType();
  DataType diag_type = diag->GetDataType();
  KERNEL_CHECK_FALSE((x_type == diag_type), KERNEL_STATUS_PARAM_INVALID,
                     "The "
                     "data type of input [%s] need be same with diagonal [%s].",
                     DTypeStr(x_type).c_str(), DTypeStr(diag_type).c_str());
  switch (x_type) {
    SET_DIAG_CASE(DT_INT8, int8_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_INT16, int16_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_INT32, int32_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_INT64, int64_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_UINT8, uint8_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_UINT16, uint16_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_UINT32, uint32_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_UINT64, uint64_t, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_FLOAT16, Eigen::half, x, diag, k_addr, y, x_size,
                  num_diags, lower_diag, upper_diag, num_row, num_col,
                  max_diag_len, ctx)
    SET_DIAG_CASE(DT_FLOAT, float, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_DOUBLE, double, x, diag, k_addr, y, x_size, num_diags,
                  lower_diag, upper_diag, num_row, num_col, max_diag_len, ctx)
    SET_DIAG_CASE(DT_COMPLEX64, std::complex<float>, x, diag, k_addr, y, x_size,
                  num_diags, lower_diag, upper_diag, num_row, num_col,
                  max_diag_len, ctx)
    SET_DIAG_CASE(DT_COMPLEX128, std::complex<double>, x, diag, k_addr, y,
                  x_size, num_diags, lower_diag, upper_diag, num_row, num_col,
                  max_diag_len, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixSetDiagV2 kernel data type [%s] not support.",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixSetDiagV2CpuKernel::SetDiagCompute(
    Tensor *x, Tensor *diag, int32_t *k, Tensor *y, uint64_t data_size,
    int32_t num_diags, int32_t lower_diag, int32_t upper_diag, int32_t num_row,
    int32_t num_col, int32_t max_diag_len, CpuKernelContext &ctx) {
  T *x_addr = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(x_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.");
  T *diag_addr = reinterpret_cast<T *>(diag->GetData());
  KERNEL_CHECK_NULLPTR(diag_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.");
  T *y_addr = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(y_addr, KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed.");

  int64_t data_num = data_size / sizeof(T);
  int64_t batch_size = num_diags * max_diag_len;
  int64_t matrix_size = num_row * num_col;
  int64_t matrix_num = data_num / matrix_size;
  if (data_num <= 64 * 1024) {
    std::memcpy(y_addr, x_addr, data_size);
    for (int64_t i = 0; i < matrix_num; i++) {
      for (int64_t m = 0; m < num_diags; m++) {
        int32_t diag_index = upper_diag - m;
        int32_t diag_len = std::min(num_row + std::min(0, diag_index),
                                    num_col - std::max(0, diag_index));
        if (diag_index >= 0) {
          for (int32_t n = 0; n < diag_len; n++) {
            int64_t y_offset = i * matrix_size + n * num_col + n + diag_index;
            int64_t diag_offset = i * batch_size + m * max_diag_len + n;
            *(y_addr + y_offset) = *(diag_addr + diag_offset);
          }
        } else {
          for (int32_t n = 0; n < diag_len; n++) {
            int64_t y_offset = i * matrix_size + (n - diag_index) * num_col + n;
            int64_t diag_offset = i * batch_size + m * max_diag_len + n;
            *(y_addr + y_offset) = *(diag_addr + diag_offset);
          }
        }
      }
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_copy = [&x_addr, &y_addr](size_t start, size_t end) {
      std::memcpy(y_addr + start, x_addr + start, (end - start) * sizeof(T));
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(
                            ctx, data_num, data_num / max_core_num, shard_copy),
                        "MatrixSetDiagV2 Compute failed.");

    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto shard_setdiag = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        for (int64_t m = 0; m < num_diags; m++) {
          int32_t diag_index = upper_diag - m;
          int32_t diag_len = std::min(num_row + std::min(0, diag_index),
                                      num_col - std::max(0, diag_index));
          if (diag_index >= 0) {
            for (int32_t n = 0; n < diag_len; n++) {
              int64_t y_offset = i * matrix_size + n * num_col + n + diag_index;
              int64_t diag_offset = i * batch_size + m * max_diag_len + n;
              *(y_addr + y_offset) = *(diag_addr + diag_offset);
            }
          } else {
            for (int32_t n = 0; n < diag_len; n++) {
              int64_t y_offset =
                  i * matrix_size + (n - diag_index) * num_col + n;
              int64_t diag_offset = i * batch_size + m * max_diag_len + n;
              *(y_addr + y_offset) = *(diag_addr + diag_offset);
            }
          }
        }
      }
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num,
                                    shard_setdiag),
        "MatrixSetDiagV2 Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixSetDiagV2, MatrixSetDiagV2CpuKernel);
}  // namespace aicpu