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
#include "sub.h"

#include <complex>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char *kSub = "Sub";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
}  // namespace
namespace aicpu {
template <typename T, int32_t RANK>
uint32_t SubCpuKernel::BroadcastCompute(TensorMap<T> &x, TensorMap<T> &y,
                                        TensorMap<T> &out, Bcast &bcast) {
  Eigen::DSizes<Eigen::DenseIndex, RANK> x_reshape;
  Eigen::DSizes<Eigen::DenseIndex, RANK> y_reshape;
  Eigen::DSizes<Eigen::DenseIndex, RANK> result_shape;
  Eigen::array<Eigen::DenseIndex, RANK> x_bcast;
  Eigen::array<Eigen::DenseIndex, RANK> y_bcast;

  for (int32_t i = 0; i < RANK; i++) {
    x_reshape[i] = bcast.x_reshape()[i];
    y_reshape[i] = bcast.y_reshape()[i];
    result_shape[i] = bcast.result_shape()[i];
    x_bcast[i] = bcast.x_bcast()[i];
    y_bcast[i] = bcast.y_bcast()[i];
  }
  out.reshape(result_shape) = x.reshape(x_reshape).broadcast(x_bcast) -
                              y.reshape(y_reshape).broadcast(y_bcast);
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t SubCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input1_tensor = ctx.Input(1);
  DataType input0_dt = input0_tensor->GetDataType();
  DataType input1_dt = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input0_dt == input1_dt), KERNEL_STATUS_INNER_ERROR,
                     "Input[x1] data type[%s] and input[x2] data type[%s] "
                     "must be same.",
                     DTypeStr(input0_dt).c_str(), DTypeStr(input1_dt).c_str());
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  auto input0_elements_num = input0_tensor->NumElements();
  TensorMap<T> input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                      input0_elements_num);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  auto input1_elements_num = input1_tensor->NumElements();
  TensorMap<T> input1(reinterpret_cast<T *>(input1_tensor->GetData()),
                      input1_elements_num);
  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  auto output_elements_num = output_tensor->NumElements();
  TensorMap<T> output(reinterpret_cast<T *>(output_tensor->GetData()),
                      output_elements_num);

  Bcast bcast(input0_shape, input1_shape);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t rank = static_cast<int32_t>(bcast.x_reshape().size());
  switch (rank) {
    case 1:
      return BroadcastCompute<T, 1>(input0, input1, output, bcast);
    case 2:
      return BroadcastCompute<T, 2>(input0, input1, output, bcast);
    case 3:
      return BroadcastCompute<T, 3>(input0, input1, output, bcast);
    case 4:
      return BroadcastCompute<T, 4>(input0, input1, output, bcast);
    default:
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t SubCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Sub params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x1] data type is [%s].", kSub,
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
    case DT_UINT8:
      ret = DoCompute<uint8_t>(ctx);
      break;
    case DT_INT8:
      ret = DoCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      ret = DoCompute<uint16_t>(ctx);
      break;
    case DT_INT16:
      ret = DoCompute<int16_t>(ctx);
      break;
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      ret = DoCompute<int64_t>(ctx);
      break;
    case DT_COMPLEX64:
      ret = DoCompute<complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoCompute<complex<double>>(ctx);
      break;
    default:
     KERNEL_LOG_ERROR("Unsupported input[x1] data type[%s]",
                      DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kSub, SubCpuKernel);
}  // namespace aicpu