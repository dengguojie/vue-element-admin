/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#ifndef AICPU_KERNELS_PINVERSE_CC_
#define AICPU_KERNELS_PINVERSE_CC_

#include "pinverse.h"
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include <iostream>

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kPinverse = "Pinverse";
const uint32_t SHAPEDIM = 2;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t PinverseCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto rcond = (ctx.GetAttr("rcond") == nullptr) ? 0 : (ctx.GetAttr("rcond")->GetFloat());
  auto input_tensor = ctx.Input(0);
  auto input_tensor_shape = input_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(input_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[x] must be a matrix");

  auto input_shape = input_tensor_shape->GetDimSizes();
  KERNEL_CHECK_FALSE((input_shape.size() == SHAPEDIM), KERNEL_STATUS_PARAM_INVALID, "Input[x] must be 2D.");
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input(reinterpret_cast<T *>(input_tensor->GetData()),
                   input_shape[0], input_shape[1]);
  auto svd = input.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &singularValues = svd.singularValues();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> singularValuesInv(input.cols(), input.rows());
  for (uint32_t i = 0; i < input.cols(); ++i) {
    for (uint32_t j = 0; j < input.rows(); ++j) {
      singularValuesInv(i, j) = 0;
    }
  }

  // defination of output tensor
  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((output_shape.size() == SHAPEDIM), KERNEL_STATUS_PARAM_INVALID, "Output[x] must be 2D.");
  MatrixMap output(reinterpret_cast<T *>(output_tensor->GetData()),
                    output_shape[0], output_shape[1]);
  
  // calculate S+
  for (unsigned int i = 0; i < singularValues.size(); ++i) {
    if (singularValues(i) > rcond) {
      singularValuesInv(i, i) = 1.0f / singularValues(i);
    } else {
      singularValuesInv(i, i) = 0.f;
    } // if else
  } // for

  // calculate pinv (origin Eigen::MatrixXf)
  output = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
  return KERNEL_STATUS_OK;
}

uint32_t PinverseCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Pinverse input and output number failed.");
  Tensor *input0_tensor = ctx.Input(0);
  auto input_x_type = input0_tensor->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch(input_x_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input x data type[%s]",
                       DTypeStr(input0_tensor->GetDataType()).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kPinverse, PinverseCpuKernel);
}  // namespace aicpu
#endif  // AICPU_KERNELS_PINVERSE_CC_
