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

#include "update_cache_kernels.h"
#include <securec.h>
#include <chrono>
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *UPDATE_CACHE = "UpdateCache";
}

namespace aicpu {
template <typename T>
uint32_t UpdateCacheTask(std::vector<Tensor *> &inputs_,
                         std::vector<Tensor *> &outputs_, int64_t batch_size_,
                         int64_t update_length_, int type_size) {
  auto start = std::chrono::high_resolution_clock::now();

  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR(
        "UpdateCacheKernel::UpdateCacheTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  char *input_x = reinterpret_cast<char *>(inputs_[0]->GetData());
  KERNEL_CHECK_NULLPTR(input_x, KERNEL_STATUS_PARAM_INVALID,
                       "Get input_x data failed")
  uint64_t input_x_size = inputs_[0]->GetDataSize();
  T *indices = reinterpret_cast<T *>(inputs_[1]->GetData());
  KERNEL_CHECK_NULLPTR(indices, KERNEL_STATUS_PARAM_INVALID,
                       "Get indices data failed")
  char *update = reinterpret_cast<char *>(inputs_[2]->GetData());
  KERNEL_CHECK_NULLPTR(update, KERNEL_STATUS_PARAM_INVALID,
                       "Get update data failed")
  uint64_t update_size = inputs_[2]->GetDataSize();
  T max_num = *reinterpret_cast<T *>(inputs_[3]->GetData());

  int64_t one_length_size = type_size * update_length_;
  KERNEL_LOG_INFO("UpdateCache one_length_size %d.", one_length_size);

  for (int64_t i = 0; i < batch_size_; ++i) {
    if (indices[i] < 0 || indices[i] >= max_num) {
      continue;
    }

    if (static_cast<uint64_t>(indices[i] * one_length_size + one_length_size) > input_x_size) {
      KERNEL_LOG_ERROR(
          "input error, indices[%lld]:%lld, one_length_size:%lld, "
          "input_x_size:%llu.",
          i, indices[i], one_length_size, input_x_size);
      return KERNEL_STATUS_INNER_ERROR;
    }

    if (static_cast<uint64_t>(i * one_length_size + one_length_size) > update_size) {
      KERNEL_LOG_ERROR(
          "input error, i:%lld, one_length_size:%lld, update_size:%llu.", i,
          one_length_size, update_size);
      return KERNEL_STATUS_INNER_ERROR;
    }

    char *tmp = update + i * one_length_size;
    int ret = memcpy_s(input_x + indices[i] * one_length_size, one_length_size,
                       tmp, one_length_size);
    if (ret != 0) {
      KERNEL_LOG_ERROR("UpdateCache memcpy failed, result %d.", ret);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  KERNEL_LOG_INFO(
      "UpdateCache execute %fms.",
      std::chrono::duration<double, std::milli>(end - start).count());
  return KERNEL_STATUS_OK;
}

uint32_t UpdateCacheKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                       std::vector<Tensor *> &, int64_t,
                                       int64_t, int)>>
      calls;
  calls[DT_INT32] = UpdateCacheTask<int32_t>;
  calls[DT_INT64] = UpdateCacheTask<int64_t>;

  if (calls.find(indices_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
        "UpdateCacheKernel op don't support indices tensor types: %s",
        typeid(indices_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int type_size = GetSizeByDataType(param_type_);
  return calls[indices_type_](inputs_, outputs_, batch_size_, update_length_,
                              type_size);
}

uint32_t UpdateCacheKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("UpdateCacheKernel::GetInputAndCheck start!");

  // get input Tensors
  const int num_input = 4;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "UpdateCacheKernel::GetInputAndCheck: get input tensor[%d] failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int num_output = 1;
  for (int i = 0; i < num_output; ++i) {
    Tensor *tensor = ctx.Output(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "UpdateCacheKernel::GetInputAndCheck: get output tensor[%d] failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  indices_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
  KERNEL_LOG_INFO("UpdateCacheKernel::GetInputAndCheck success!");

  std::shared_ptr<TensorShape> param_shape = ctx.Input(0)->GetTensorShape();
  std::shared_ptr<TensorShape> indices_shape = ctx.Input(1)->GetTensorShape();
  std::shared_ptr<TensorShape> update_shape = ctx.Input(2)->GetTensorShape();

  batch_size_ = ctx.Input(1)->NumElements();
  if (batch_size_ <= 0) {
    KERNEL_LOG_ERROR("Get input tensor[1] element number:%lld failed",
                     batch_size_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  update_size_ = ctx.Input(2)->NumElements();

  update_length_ = update_size_ / batch_size_;
  return KERNEL_STATUS_OK;
}

uint32_t UpdateCacheKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("UpdateCacheKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("UpdateCacheKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("UpdateCacheKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(UPDATE_CACHE, UpdateCacheKernel);
}  // namespace aicpu
