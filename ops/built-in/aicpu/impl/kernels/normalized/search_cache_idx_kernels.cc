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

#include "search_cache_idx_kernels.h"
#include <chrono>
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *SEARCH_CACHE_IDX = "SearchCacheIdx";
}

namespace aicpu {
template <typename T>
uint32_t SearchCacheIdxTask(std::vector<Tensor *> &inputs_,
                            std::vector<Tensor *> &outputs_,
                            int64_t &batch_size_, int64_t &hashmap_length_) {
  auto start = std::chrono::high_resolution_clock::now();

  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR(
        "SearchCacheIdxKernel::SearchCacheIdxTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  HashmapEntry<T> *hashmap =
      reinterpret_cast<HashmapEntry<T> *>(inputs_[0]->GetData());
  T *input_indices = reinterpret_cast<T *>(inputs_[1]->GetData());
  int64_t step_ = *reinterpret_cast<T *>(inputs_[2]->GetData());
  int64_t emb_max_num_ = *reinterpret_cast<T *>(inputs_[3]->GetData());
  int64_t cache_max_num_ = *reinterpret_cast<T *>(inputs_[4]->GetData());

  T *output_cache_idx = reinterpret_cast<T *>(
      outputs_[0]->GetData());  // 16000*39, if miss, cacheIdx=-1
  T *output_miss_idx =
      reinterpret_cast<T *>(outputs_[1]->GetData());  // 16000*39, default -1
  T *output_miss_emb_idx =
      reinterpret_cast<T *>(outputs_[2]->GetData());  // 16000*39, default -1

  float total_count = 0;
  float total_hit = 0;
  int count_size = 1;

  for (int64_t i = 0; i < batch_size_; ++i) {
    if (input_indices[i] >= emb_max_num_) {
      output_miss_idx[i] = -1;
      output_cache_idx[i] = cache_max_num_;
      output_miss_emb_idx[i] = -1;
      continue;
    }

    T key = input_indices[i];
    T entry = HashFunc(key, hashmap_length_);
    T tmp_entry = entry;
    int count = 1;
    count_size += 1;
    while ((!hashmap[tmp_entry].IsEmpty() && !hashmap[tmp_entry].IsKey(key))) {
      tmp_entry = (tmp_entry + 1) % hashmap_length_;
      count++;
    }
    total_count += count;
    if (hashmap[tmp_entry].IsEmpty()) {
      output_miss_idx[i] = i;
      output_miss_emb_idx[i] = key;
      output_cache_idx[i] = -1;
    }

    else {
      output_miss_idx[i] = -1;
      output_cache_idx[i] = hashmap[tmp_entry].value;
      hashmap[tmp_entry].step = step_;
      output_miss_emb_idx[i] = -1;
      total_hit += 1;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  KERNEL_LOG_INFO("SearchCacheIdx avg count %f.", total_count / count_size);
  KERNEL_LOG_INFO("SearchCacheIdx hit rate %f.", total_hit / count_size);
  KERNEL_LOG_INFO(
      "SearchCacheIdx execute %fms.",
      std::chrono::duration<double, std::milli>(end - start).count());
  return KERNEL_STATUS_OK;
}

uint32_t SearchCacheIdxKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                       std::vector<Tensor *> &, int64_t &,
                                       int64_t &)>>
      calls;
  calls[DT_INT32] = SearchCacheIdxTask<int32_t>;
  calls[DT_INT64] = SearchCacheIdxTask<int64_t>;

  if (calls.find(param_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
        "SearchCacheIdxKernel op don't support input tensor types: %s",
        typeid(param_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls[param_type_](inputs_, outputs_, batch_size_, hashmap_length_);
}

uint32_t SearchCacheIdxKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SearchCacheIdxKernel::GetInputAndCheck start!");

  // get input Tensors
  const int num_input = 5;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "SearchCacheIdxKernel::GetInputAndCheck: get input tensor[%d] failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int num_output = 3;
  for (int i = 0; i < num_output; ++i) {
    Tensor *tensor = ctx.Output(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "SearchCacheIdxKernel::GetInputAndCheck: get output tensor[%d] "
          "failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  KERNEL_LOG_INFO("SearchCacheIdxKernel::GetInputAndCheck success!");

  std::shared_ptr<TensorShape> hashmap_shape = ctx.Input(0)->GetTensorShape();
  std::shared_ptr<TensorShape> emb_idx_shape = ctx.Input(1)->GetTensorShape();
  if (hashmap_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
        "SearchCacheIdxKernel::GetInputAndCheck: only support hashmap rank 2, "
        "but got %d",
        hashmap_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  hashmap_length_ = hashmap_shape->GetDimSize(0);
  if (hashmap_shape->GetDimSize(1) != 4) {
    KERNEL_LOG_ERROR(
        "SearchCacheIdxKernel::GetInputAndCheck: only support hashmap shape "
        "(n, 4), but got (%d, %d)",
        hashmap_length_, hashmap_shape->GetDimSize(1));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (int i = 0; i < emb_idx_shape->GetDims(); ++i) {
    batch_size_ *= emb_idx_shape->GetDimSize(i);
  }
  return KERNEL_STATUS_OK;
}

uint32_t SearchCacheIdxKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SearchCacheIdxKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }
  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SearchCacheIdxKernel::Compute failed");
    return res;
  }
  KERNEL_LOG_INFO("SearchCacheIdxKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(SEARCH_CACHE_IDX, SearchCacheIdxKernel);
}  // namespace aicpu
