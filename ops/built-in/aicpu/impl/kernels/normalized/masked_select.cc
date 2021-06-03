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
#include "masked_select.h"

#include "Eigen/Core"
#include "securec.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMaskedSelectInputNum = 2;
constexpr uint32_t kMaskedSelectOutputNum = 1;
const char *kMaskedSelect = "MaskedSelect";
}

namespace aicpu {
uint32_t MaskedSelectCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaskedSelectInputNum, kMaskedSelectOutputNum),
                      "[%s] check params failed.", kMaskedSelect);

  // choose compute function depend on dataType
  auto data_type0 =
      static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto data_type1 =
      static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  auto data_type2 =
      static_cast<DataType>(ctx.Input(kFirstOutputIndex)->GetDataType());
  if (data_type1 != DT_BOOL) {
      KERNEL_LOG_ERROR("[%s] Data type of mask requires bool, but got data type [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type0 != data_type2) {
      KERNEL_LOG_ERROR("[%s] Data type of x and y requires same, but got data type [%s] and [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type2).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (data_type0) {
    case DT_FLOAT16:
      return MaskedSelectCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MaskedSelectCompute<float>(ctx);
    case DT_DOUBLE:
      return MaskedSelectCompute<double>(ctx);
    case DT_INT8:
      return MaskedSelectCompute<int8_t>(ctx);
    case DT_INT16:
      return MaskedSelectCompute<int16_t>(ctx);
    case DT_INT32:
      return MaskedSelectCompute<int32_t>(ctx);
    case DT_INT64:
      return MaskedSelectCompute<int64_t>(ctx);
    case DT_UINT8:
      return MaskedSelectCompute<uint8_t>(ctx);
    case DT_UINT16:
      return MaskedSelectCompute<uint16_t>(ctx);
    case DT_UINT32:
      return MaskedSelectCompute<uint32_t>(ctx);
    case DT_UINT64:
      return MaskedSelectCompute<uint64_t>(ctx);
    case DT_BOOL:
      return MaskedSelectCompute<bool>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t MaskedSelectCpuKernel::MaskedSelectCompute(CpuKernelContext &ctx) {
  T *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_data[0] failed.", kMaskedSelect);
  bool *mask = reinterpret_cast<bool *>(ctx.Input(1)->GetData());
  KERNEL_CHECK_NULLPTR(mask, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_data[1] failed.", kMaskedSelect);
  T *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output_data[0] failed.", kMaskedSelect);

  auto input_shape_a = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_shape_b = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape;
  auto ret = GetBroadcastShape(input_shape_a, input_shape_b, output_shape);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "Shape of x and mask can't be broadcast.");
  int64_t tensor_size = 1;
  for (const int64_t &d : output_shape) {
    tensor_size *= d;
  }
  int64_t j = 0;
  BroadcastIterator iter(input_shape_a, input_shape_b, output_shape);
  iter.SetPos(0);
  for (int64_t i = 0; i < tensor_size; ++i) {
    if (mask[iter.GetInputPosB()]) {
      y[j++] = x[iter.GetInputPosA()];
    }
    iter.GenNextPos();
  }
  ctx.Output(0)->GetTensorShape()->SetDimSizes({j});
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kMaskedSelect, MaskedSelectCpuKernel);
}  // namespace aicpu
