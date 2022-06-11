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

#include "conjugate_transpose.h"

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
constexpr int64_t dim_2 = 2;
constexpr int64_t dim_3 = 3;
constexpr int64_t dim_4 = 4;
constexpr int64_t dim_5 = 5;
constexpr int64_t dim_6 = 6;
constexpr int64_t dim_7 = 7;
const char *kConjugateTranspose = "ConjugateTranspose";

#define CONJUGATETRANSPOSE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                    \
    uint32_t result = ConjugateTransposeCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("ConjugateTranspose kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

#define CONJUGATETRANSPOSE_COMPUTE_CASE3(input_dims, perm_nd) \
  for (size_t i = 0; i < (input_dims); ++i) {                 \
    (perm_nd)[i] = perm.at(i);                                \
  }

#define CONJUGATETRANSPOSE_COMPUTE_DIM2(input_data, shape_x, output_data,     \
                                        shape_y)                              \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>,          \
                           Eigen::Aligned>                                    \
      Eigen_Tensor_ND;                                                        \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1));   \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1)); \
  Eigen::array<Eigen::DenseIndex, dim_2> perm_2d;                             \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_2, perm_2d)                            \
  output_ND = input_ND.shuffle(perm_2d).conjugate();

#define CONJUGATETRANSPOSE_COMPUTE_DIM3(input_data, shape_x, output_data,    \
                                        shape_y)                             \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_3, Eigen::RowMajor>,         \
                           Eigen::Aligned>                                   \
      Eigen_Tensor_ND;                                                       \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1),   \
                           (shape_x).at(2));                                 \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1), \
                            (shape_y).at(2));                                \
  Eigen::array<Eigen::DenseIndex, dim_3> perm_3d;                            \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_3, perm_3d)                           \
  output_ND = input_ND.shuffle(perm_3d).conjugate();

#define CONJUGATETRANSPOSE_COMPUTE_DIM4(input_data, shape_x, output_data,    \
                                        shape_y)                             \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_4, Eigen::RowMajor>,         \
                           Eigen::Aligned>                                   \
      Eigen_Tensor_ND;                                                       \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1),   \
                           (shape_x).at(dim_2), (shape_x).at(dim_3));        \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1), \
                            (shape_y).at(dim_2), (shape_y).at(dim_3));       \
  Eigen::array<Eigen::DenseIndex, dim_4> perm_4d;                            \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_4, perm_4d)                           \
  output_ND = input_ND.shuffle(perm_4d).conjugate();

#define CONJUGATETRANSPOSE_COMPUTE_DIM5(input_data, shape_x, output_data,    \
                                        shape_y)                             \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_5, Eigen::RowMajor>,         \
                           Eigen::Aligned>                                   \
      Eigen_Tensor_ND;                                                       \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1),   \
                           (shape_x).at(dim_2), (shape_x).at(dim_3),         \
                           (shape_x).at(dim_4));                             \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1), \
                            (shape_y).at(dim_2), (shape_y).at(dim_3),        \
                            (shape_y).at(dim_4));                            \
  Eigen::array<Eigen::DenseIndex, dim_5> perm_5d;                            \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_5, perm_5d)                           \
  output_ND = input_ND.shuffle(perm_5d).conjugate();

#define CONJUGATETRANSPOSE_COMPUTE_DIM6(input_data, shape_x, output_data,    \
                                        shape_y)                             \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_6, Eigen::RowMajor>,         \
                           Eigen::Aligned>                                   \
      Eigen_Tensor_ND;                                                       \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1),   \
                           (shape_x).at(dim_2), (shape_x).at(dim_3),         \
                           (shape_x).at(dim_4), (shape_x).at(dim_5));        \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1), \
                            (shape_y).at(dim_2), (shape_y).at(dim_3),        \
                            (shape_y).at(dim_4), (shape_y).at(dim_5));       \
  Eigen::array<Eigen::DenseIndex, dim_6> perm_6d;                            \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_6, perm_6d)                           \
  output_ND = input_ND.shuffle(perm_6d).conjugate();

#define CONJUGATETRANSPOSE_COMPUTE_DIM7(input_data, shape_x, output_data,    \
                                        shape_y)                             \
  typedef Eigen::TensorMap<Eigen::Tensor<T, dim_7, Eigen::RowMajor>,         \
                           Eigen::Aligned>                                   \
      Eigen_Tensor_ND;                                                       \
  Eigen_Tensor_ND input_ND((input_data), (shape_x).at(0), (shape_x).at(1),   \
                           (shape_x).at(dim_2), (shape_x).at(dim_3),         \
                           (shape_x).at(dim_4), (shape_x).at(dim_5),         \
                           (shape_x).at(dim_6));                             \
  Eigen_Tensor_ND output_ND((output_data), (shape_y).at(0), (shape_y).at(1), \
                            (shape_y).at(dim_2), (shape_y).at(dim_3),        \
                            (shape_y).at(dim_4), (shape_y).at(dim_5),        \
                            (shape_y).at(dim_6));                            \
  Eigen::array<Eigen::DenseIndex, dim_7> perm_7d;                            \
  CONJUGATETRANSPOSE_COMPUTE_CASE3(dim_7, perm_7d)                           \
  output_ND = input_ND.shuffle(perm_7d).conjugate();
}  // namespace

namespace aicpu {
uint32_t ConjugateTransposeCpuKernel::GetConjugateTransposeValue(
    Tensor *tensor, std::vector<int64_t> &value) {
  auto type = tensor->GetDataType();
  if (type == DT_INT32) {
    auto data = reinterpret_cast<int32_t *>(tensor->GetData());
    for (unsigned int i = 0; i < tensor->NumElements(); i++) {
      value.push_back(static_cast<int64_t>(*(data + i)));
    }
  } else if (type == DT_INT64) {
    auto data = reinterpret_cast<int64_t *>(tensor->GetData());
    for (unsigned int i = 0; i < tensor->NumElements(); i++) {
      value.push_back(*(data + i));
    }
  } else {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ConjugateTransposeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.",
                      kConjugateTranspose);
  KERNEL_HANDLE_ERROR(ConjugateTransposeParamCheck(ctx),
                      "[%s] check params failed.", kConjugateTranspose);
  auto x_type = ctx.Input(0)->GetDataType();
  switch (x_type) {
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_BOOL, bool, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_INT8, std::int8_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_INT16, std::int16_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_INT32, std::int32_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_INT64, std::int64_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_UINT8, std::uint8_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_UINT16, std::uint16_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_UINT32, std::uint32_t, ctx)
    CONJUGATETRANSPOSE_COMPUTE_CASE(DT_UINT64, std::uint64_t, ctx)
    default:
      KERNEL_LOG_ERROR("ConjugateTranspose kernel data type [%s] not support.",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t ConjugateTransposeCpuKernel::ConjugateTransposeParamCheck(
    CpuKernelContext &ctx) {
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_perm =
      ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto perm_tensor = ctx.Input(1);
  auto y_tensor = ctx.Output(0);
  KERNEL_CHECK_FALSE(
      (shape_perm.size() == 1), KERNEL_STATUS_PARAM_INVALID,"Expected perm to "
      "be 1-D tensors , but got [%zu]-D tensors.", shape_x.size())
  KERNEL_CHECK_FALSE(
      (perm_tensor->NumElements() == (unsigned int)shape_x.size()),
      KERNEL_STATUS_PARAM_INVALID, "Expected the size of perm to be [%zu], but "
      "got [%zu].", shape_x.size(), perm_tensor->NumElements())
  KERNEL_CHECK_FALSE(
      (GetConjugateTransposeValue(perm_tensor, perm) == KERNEL_STATUS_OK),
      KERNEL_STATUS_PARAM_INVALID, "perm must be either int32 or int64, "
      "but got [%s].", DTypeStr(perm_tensor->GetDataType()).c_str())
  KERNEL_CHECK_FALSE(
      (shape_x.size() > 1), KERNEL_STATUS_PARAM_INVALID,
      "Expected the dimension of x to be greater than 1-D, but got [%zu].",
      shape_x.size())
  std::vector<int64_t> shape_y;
  for (size_t i = 0; i < shape_x.size(); ++i) {
    int64_t perm_value = perm.at(i);
    if (shape_x.at(i) == 0) {
      KERNEL_CHECK_FALSE(
          (perm_value == 0), KERNEL_STATUS_PARAM_INVALID,
          "Expected perm[%zu] == 0 (got %zu), when x shape[%zu] == 0.", i,
          perm_value, i)
    } else {
      KERNEL_CHECK_FALSE(
          (0 <= perm_value && perm_value <= (unsigned int)shape_x.size() - 1),
          KERNEL_STATUS_PARAM_INVALID,
          "Expected perm[%zu] in [0, %zu], but got %zu.", i, shape_x.size(),
          perm_value)
    }
    int64_t temp_value = 0;
    for (size_t j = 0; j < shape_x.size(); ++j) {
      if ((unsigned int)perm.at(j) == i) {
        break;
      } else {
        temp_value = j + 1;
        KERNEL_CHECK_FALSE((temp_value < (unsigned int)shape_x.size()),
                           KERNEL_STATUS_PARAM_INVALID,
                           "Expected perm value is unique.")
      }
    }
    shape_y.push_back(shape_x.at(perm_value));
  }
  y_tensor->GetTensorShape()->SetDimSizes(shape_y);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConjugateTransposeCpuKernel::ConjugateTransposeCompute(
    CpuKernelContext &ctx) {
  auto x_data = ctx.Input(0)->GetData();
  auto y_data = ctx.Output(0)->GetData();
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<T *>(x_data);
  auto output_data = reinterpret_cast<T *>(y_data);
  int64_t input_dims = shape_x.size();
  switch (input_dims) {
    case dim_2: {
      CONJUGATETRANSPOSE_COMPUTE_DIM2(input_data, shape_x, output_data, shape_y)
      break;
    }
    case dim_3: {
      CONJUGATETRANSPOSE_COMPUTE_DIM3(input_data, shape_x, output_data, shape_y)
      break;
    }
    case dim_4: {
      CONJUGATETRANSPOSE_COMPUTE_DIM4(input_data, shape_x, output_data, shape_y)
      break;
    }
    case dim_5: {
      CONJUGATETRANSPOSE_COMPUTE_DIM5(input_data, shape_x, output_data, shape_y)
      break;
    }
    case dim_6: {
      CONJUGATETRANSPOSE_COMPUTE_DIM6(input_data, shape_x, output_data, shape_y)
      break;
    }
    case dim_7: {
      CONJUGATETRANSPOSE_COMPUTE_DIM7(input_data, shape_x, output_data, shape_y)
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] : Unhandled input dimensions [%zu].",
                       kConjugateTranspose, input_dims);
      return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kConjugateTranspose, ConjugateTransposeCpuKernel);
}  // namespace aicpu
