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
#include "matmul.h"

#include <complex>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char *kMatmul = "MatMul";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t MatMulCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(input0_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[x1] must be a matrix")

  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(input1_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[x2] must be a matrix")

  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input1_data_type = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE(
      (input0_data_type == input1_data_type),
      KERNEL_STATUS_PARAM_INVALID,
      "Input[x1] data type[%s] and input[x2] data type[%s] must be same",
      DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str())

  auto transpose_x1 = ctx.GetAttr("transpose_x1")->GetBool();
  auto transpose_x2 = ctx.GetAttr("transpose_x2")->GetBool();
  KERNEL_LOG_DEBUG("%s Attr[transpose_x1] value[%d], "
                   "Attr[transpose_x2] value[%d].", kMatmul,
                   transpose_x1, transpose_x2);
  int32_t x1_dim = transpose_x1 ? 0 : 1;
  int32_t x2_dim = transpose_x2 ? 1 : 0;
  KERNEL_CHECK_FALSE((input0_tensor_shape->GetDimSize(x1_dim) ==
                      input1_tensor_shape->GetDimSize(x2_dim)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Matrix size incompatible, input[x1] dim[%d] value[%lld], "
                     "input[x2] dim[%d] value[%lld]",
                     x1_dim, input0_tensor_shape->GetDimSize(x1_dim),
                     x2_dim, input1_tensor_shape->GetDimSize(x2_dim))

  auto input0_shape = input0_tensor_shape->GetDimSizes();
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                   input0_shape[0], input0_shape[1]);

  auto input1_shape = input1_tensor_shape->GetDimSizes();
  MatrixMap input1(reinterpret_cast<T *>(input1_tensor->GetData()),
                   input1_shape[0], input1_shape[1]);

  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap output(reinterpret_cast<T *>(output_tensor->GetData()),
                   output_shape[0], output_shape[1]);
  if (transpose_x1) {
    if (transpose_x2) {
      output = input0.transpose() * input1.transpose();
    } else {
      output = input0.transpose() * input1;
    }
  } else {
    if (transpose_x2) {
      output = input0 * input1.transpose();
    } else {
      output = input0 * input1;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"transpose_x1", "transpose_x2"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Check Greater params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x1] data type is [%s].", kMatmul,
                   DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[x1] data type[%s]",
                       DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kMatmul, MatMulCpuKernel);
}  // namespace aicpu