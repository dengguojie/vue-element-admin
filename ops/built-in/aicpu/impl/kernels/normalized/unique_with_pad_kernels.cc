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

#include "unique_with_pad_kernels.h"

#include <memory.h>
#include <ctime>
#include <unordered_map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"

using std::string;

namespace {
const char *UNIQUE = "UniqueWithPad";
}

namespace aicpu {
uint32_t UniqueWithPadCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("UniqueWithPadCpuKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("UniqueWithPadCpuKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("UniqueWithPadCpuKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}

uint32_t UniqueWithPadCpuKernel::DoCompute() {
  uint32_t res;
  switch (matrix_type_) {
    case DT_INT32: {
      res = UniqueWithPadTask<int32_t>();
      break;
    }
    case DT_INT64: {
      res = UniqueWithPadTask<int64_t>();
      break;
    }
    default: {
      KERNEL_LOG_ERROR("UniqueWithPad op don't support input tensor types: %s",
                       typeid(matrix_type_).name());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  return res;
}

template <typename T>
uint32_t UniqueWithPadCpuKernel::UniqueWithPadTask() {
  clock_t start, end;
  start = clock();
  T *a = reinterpret_cast<T *>(input_tensor_->GetData());
  T padding = *static_cast<T *>(input_padding_->GetData());
  T *out = reinterpret_cast<T *>(output_values_->GetData());
  T *idx_vec = reinterpret_cast<T *>(output_indices_->GetData());
  for (int64_t i = 0; i < p_size_; ++i) {
    out[i] = padding;
  }
  std::unordered_map<T, int> uniq;
  uniq.reserve(2 * p_size_);
  for (int64_t i = 0, j = 0; i < p_size_; ++i) {
    auto it = uniq.emplace(a[i], j);
    idx_vec[i] = it.first->second;
    if (it.second) {
      ++j;
    }
  }
  for (const auto &it : uniq) {
    out[it.second] = it.first;
  }
  end = clock();
  KERNEL_LOG_INFO("UniqueWithPad execute %f ms.",
                  (float)(end - start) * 1000 / CLOCKS_PER_SEC);
  return KERNEL_STATUS_OK;
}

uint32_t UniqueWithPadCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck start!! ");

  // get input_tensor
  input_tensor_ = ctx.Input(0);
  if (input_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("get input:0 failed");
    KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck failed!! ");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
  int32_t input_rank = input_shape->GetDims();
  for (int32_t i = 0; i < input_rank; ++i) {
    p_size_ *= input_shape->GetDimSize(i);
  }
  matrix_type_ = static_cast<DataType>(input_tensor_->GetDataType());

  // get padding
  input_padding_ = ctx.Input(1);
  if (input_padding_ == nullptr) {
    KERNEL_LOG_ERROR("get input:1 failed");
    KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck failed!! ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get output
  output_values_ = ctx.Output(0);
  if (output_values_ == nullptr) {
    KERNEL_LOG_ERROR("get output:0 failed");
    KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck failed!! ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  output_indices_ = ctx.Output(1);
  if (output_indices_ == nullptr) {
    KERNEL_LOG_ERROR("get output:1 failed");
    KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck failed!! ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("UniqueWithPadCpuKernel::GetInputAndCheck success!! ");

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(UNIQUE, UniqueWithPadCpuKernel);
}  // namespace aicpu