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
#include "matrix_triangular_solve.h"

#include <complex>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char *kMatrixTriangularSolve = "MatrixTriangularSolve";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t MatrixTriangularSolveCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();

  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input1_data_type = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE(
      (input0_data_type == input1_data_type),
      KERNEL_STATUS_PARAM_INVALID,
      "Input[matrix] data type[%s] and input[rhs] data type[%s] must be same",
      DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str())

  KERNEL_CHECK_FALSE((IsSquareMatrix(input0_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[matrix] must be a square matrix")
  KERNEL_CHECK_FALSE((IsMatrix(input1_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[rhs] must be a matrix")
  KERNEL_CHECK_FALSE((input0_tensor_shape->GetDimSize(0) ==
                      input1_tensor_shape->GetDimSize(0)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input matrix and rhs are incompatible")

  auto lower = ctx.GetAttr("lower")->GetBool();
  auto adjoint = ctx.GetAttr("adjoint")->GetBool();
  auto input0_shape = input0_tensor_shape->GetDimSizes();
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                   input0_shape[0], input0_shape[1]);

  using RealScalar = typename Eigen::NumTraits<T>::Real;
  RealScalar pivot = input0.diagonal().cwiseAbs().minCoeff();
  KERNEL_CHECK_FALSE((pivot > RealScalar(0)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input matrix is not invertible")

  auto input1_shape = input1_tensor_shape->GetDimSizes();
  MatrixMap input1(reinterpret_cast<T *>(input1_tensor->GetData()),
                   input1_shape[0], input1_shape[1]);
  auto output_tensor = ctx.Output(0);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap output(reinterpret_cast<T *>(output_tensor->GetData()),
                   output_shape[0], output_shape[1]);
  if (lower) {
    auto triangle = input0.template triangularView<Eigen::Lower>();
    if (adjoint) {
      output.noalias() = triangle.adjoint().solve(input1);
    } else {
      output.noalias() = triangle.solve(input1);
    }
  } else {
    auto triangle = input0.template triangularView<Eigen::Upper>();
    if (adjoint) {
      output.noalias() = triangle.adjoint().solve(input1);
    } else {
      output.noalias() = triangle.solve(input1);
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatrixTriangularSolveCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"lower", "adjoint"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Check Greater params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[matrix] data type is [%s].", kMatrixTriangularSolve,
                   DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_COMPLEX64:
      ret = DoCompute<complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoCompute<complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[matrix] data type[%s]",
                       DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kMatrixTriangularSolve, MatrixTriangularSolveCpuKernel);
}  // namespace aicpu