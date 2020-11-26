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

#include "cache_swap_table_kernels.h"
#include <securec.h>
#include <chrono>
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *CACHE_SWAP_TABLE = "CacheSwapTable";
}

namespace aicpu {

template <typename T>
uint32_t CacheSwapTableTask(std::vector<Tensor *> &inputs_,
                            std::vector<Tensor *> &outputs_,
                            int64_t batch_size_, int64_t output_size_,
                            int64_t one_line_col_, int type_size) {
  auto start = std::chrono::high_resolution_clock::now();

  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR(
        "CacheSwapTableCpuKernel::CacheSwapTableTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  char *cache_table = reinterpret_cast<char *>(inputs_[0]->GetData());
  T *swap_cache_idx = reinterpret_cast<T *>(inputs_[1]->GetData());
  uint64_t swap_cache_idx_size = inputs_[1]->GetDataSize();
  char *miss_value = reinterpret_cast<char *>(inputs_[2]->GetData());

  char *old_value = reinterpret_cast<char *>(outputs_[0]->GetData());

  auto ret = memset_s(old_value, output_size_ * type_size, 0x00,
                      output_size_ * type_size);
  if (ret != EOK) {
    KERNEL_LOG_ERROR("memset_s failed, result[%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int64_t single_copy_size = type_size * one_line_col_;

  if (swap_cache_idx_size < static_cast<uint64_t>(batch_size_)) {
    KERNEL_LOG_ERROR(
        "swap_cache_idx_size:%llu must be less than batch_size_%lld",
        swap_cache_idx_size, batch_size_);
    return KERNEL_STATUS_INNER_ERROR;
  }

  for (int64_t i = 0; i < batch_size_; ++i) {
    if (swap_cache_idx[i] < 0) {
      continue;
    }

    int ret = memcpy_s(old_value + i * single_copy_size, single_copy_size,
                       cache_table + swap_cache_idx[i] * single_copy_size,
                       single_copy_size);
    if (ret != 0) {
      KERNEL_LOG_ERROR("CacheSwapTable memcpy failed, result %d.", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
    ret = memcpy_s(cache_table + swap_cache_idx[i] * single_copy_size,
                   single_copy_size, miss_value + i * single_copy_size,
                   single_copy_size);
    if (ret != 0) {
      KERNEL_LOG_ERROR("CacheSwapTable memcpy failed, result %d.", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  KERNEL_LOG_INFO(
      "CacheSwapTable execute %fms.",
      std::chrono::duration<double, std::milli>(end - start).count());
  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapTableCpuKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                       std::vector<Tensor *> &, int64_t &,
                                       int64_t &, int64_t &, int &)>>
      calls;
  calls[DT_INT32] = CacheSwapTableTask<int32_t>;
  calls[DT_INT64] = CacheSwapTableTask<int64_t>;

  if (calls.find(indices_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
        "CacheSwapTableCpuKernel op don't support indices tensor types: %s",
        typeid(indices_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int type_size = GetSizeByDataType(param_type_);
  return calls[indices_type_](inputs_, outputs_, batch_size_, output_size_,
                              one_line_col_, type_size);
}

uint32_t CacheSwapTableCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CacheSwapTableCpuKernel::GetInputAndCheck start!");

  // get input Tensors
  const int num_input = 3;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "CacheSwapTableCpuKernel::GetInputAndCheck: get input tensor[%d] failed",
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
          "CacheSwapTableCpuKernel::GetInputAndCheck: get output tensor[%d] "
          "failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  indices_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
  KERNEL_LOG_INFO("CacheSwapTableCpuKernel::GetInputAndCheck success!");

  std::shared_ptr<TensorShape> cache_table_shape =
      ctx.Input(0)->GetTensorShape();
  std::shared_ptr<TensorShape> indices_shape = ctx.Input(1)->GetTensorShape();

  for (int i = 1; i < cache_table_shape->GetDims(); ++i) {
    KERNEL_CHECK_ASSIGN_64S_MULTI(one_line_col_,
                                  cache_table_shape->GetDimSize(i),
                                  one_line_col_, KERNEL_STATUS_PARAM_INVALID);
  }

  for (int i = 0; i < indices_shape->GetDims(); ++i) {
    KERNEL_CHECK_ASSIGN_64S_MULTI(batch_size_, indices_shape->GetDimSize(i),
                                  batch_size_, KERNEL_STATUS_PARAM_INVALID);
  }

  output_size_ = batch_size_ * one_line_col_;
  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapTableCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CacheSwapTableCpuKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CacheSwapTableCpuKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("CacheSwapTableCpuKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(CACHE_SWAP_TABLE, CacheSwapTableCpuKernel);
}  // namespace aicpu
