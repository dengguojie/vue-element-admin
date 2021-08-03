/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "gemm.h"

#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kGemm = "GEMM";
const uint32_t kInputNum = 5;
const uint32_t kOutputNum = 1;
}  // namespace

namespace aicpu {
uint32_t GemmCpuKernel::Check(CpuKernelContext &ctx) {
  auto a_tensor = ctx.Input(0);
  auto a_tensor_shape = a_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(a_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[a] must be a matrix")

  auto b_tensor = ctx.Input(1);
  auto b_tensor_shape = b_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(b_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[b] must be a matrix")

  auto c_tensor = ctx.Input(2);
  auto c_tensor_shape = c_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(c_tensor_shape->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID, "Input[c] must be a matrix")

  auto alpha_tensor = ctx.Input(3);
  KERNEL_CHECK_FALSE((IsScalar(alpha_tensor->GetTensorShape()->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[alpha] must be a scalar")

  auto beta_tensor = ctx.Input(4);
  KERNEL_CHECK_FALSE((IsScalar(beta_tensor->GetTensorShape()->GetDimSizes())),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[beta] must be a scalar")

  DataType input0_data_type = a_tensor->GetDataType();
  for (uint32_t i = 1; i < kInputNum; ++i) {
    KERNEL_CHECK_FALSE(
        (input0_data_type == ctx.Input(i)->GetDataType()),
        KERNEL_STATUS_PARAM_INVALID,
        "Input[a] data type[%s] and input[%u] data type[%s] must be same",
        DTypeStr(input0_data_type).c_str(), i,
        DTypeStr(ctx.Input(i)->GetDataType()).c_str())
  }

  auto transpose_a = ctx.GetAttr("transpose_a")->GetBool();
  auto transpose_b = ctx.GetAttr("transpose_b")->GetBool();
  int32_t a_dim = transpose_a ? 0 : 1;
  int32_t b_dim = transpose_b ? 1 : 0;
  KERNEL_CHECK_FALSE(
      (a_tensor_shape->GetDimSize(a_dim) == b_tensor_shape->GetDimSize(b_dim)),
      KERNEL_STATUS_PARAM_INVALID, "Matrix size incompatible")

  int32_t a_dim_remaining = 1 - a_dim;
  int32_t b_dim_remaining = 1 - b_dim;
  KERNEL_CHECK_FALSE((a_tensor_shape->GetDimSize(a_dim_remaining) ==
                      c_tensor_shape->GetDimSize(0)),
                     KERNEL_STATUS_PARAM_INVALID, "Matrix size incompatible")
  KERNEL_CHECK_FALSE((b_tensor_shape->GetDimSize(b_dim_remaining) ==
                      c_tensor_shape->GetDimSize(1)),
                     KERNEL_STATUS_PARAM_INVALID, "Matrix size incompatible")

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GemmCpuKernel::DoCompute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(Check(ctx), "Check Gemm params failed.");
  auto a_tensor = ctx.Input(0);
  auto input_a_shape = a_tensor->GetTensorShape()->GetDimSizes();
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input_a(reinterpret_cast<T *>(a_tensor->GetData()),
                    input_a_shape[0], input_a_shape[1]);

  auto b_tensor = ctx.Input(1);
  auto input_b_shape = b_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap input_b(reinterpret_cast<T *>(b_tensor->GetData()),
                    input_b_shape[0], input_b_shape[1]);

  auto c_tensor = ctx.Input(2);
  auto input_c_shape = c_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap input_c(reinterpret_cast<T *>(c_tensor->GetData()),
                    input_c_shape[0], input_c_shape[1]);

  T alpha = *static_cast<T *>(ctx.Input(3)->GetData());
  T beta = *static_cast<T *>(ctx.Input(4)->GetData());

  auto output_tensor = ctx.Output(0);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap output(reinterpret_cast<T *>(output_tensor->GetData()),
                   output_shape[0], output_shape[1]);

  auto transpose_a = ctx.GetAttr("transpose_a")->GetBool();
  auto transpose_b = ctx.GetAttr("transpose_b")->GetBool();
  if (transpose_a) {
    if (transpose_b) {
      output =
          alpha * input_a.transpose() * input_b.transpose() + beta * input_c;
    } else {
      output = alpha * input_a.transpose() * input_b + beta * input_c;
    }
  } else {
    if (transpose_a) {
      output = alpha * input_a * input_b.transpose() + beta * input_c;
    } else {
      output = alpha * input_a * input_b + beta * input_c;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t GemmCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"transpose_a", "transpose_b"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Check Gemm params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[a] data type is [%s].", kGemm,
                   DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_INT8:
      ret = DoCompute<int8_t>(ctx);
      break;
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[a] data type[%s]",
                       DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kGemm, GemmCpuKernel);
}  // namespace aicpu
