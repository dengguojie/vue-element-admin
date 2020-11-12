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

#include "cache_swap_hashmap_kernels.h"
#include <chrono>
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *CACHE_SWAP_HASHMAP = "CacheSwapHashmap";
}

namespace aicpu {
template <typename T>
int Compress(HashmapEntry<T> *entry_p, const int64_t &length, T entry) {
  T i = (entry + 1) % length, off = 1;
  int compress_count = 1;
  for (; !entry_p[i].IsEmpty(); i = (i + 1) % length, off++) {
    if (entry_p[i].tag > off) {
      entry_p[entry].key = entry_p[i].key;
      entry_p[entry].value = entry_p[i].value;
      entry_p[entry].step = entry_p[i].step;
      entry_p[entry].tag = entry_p[i].tag - off;
      entry_p[i].SetEmpty();
      off = 0;
      entry = i;
    }
    compress_count++;
  }
  return compress_count;
}

template <typename T>
uint32_t CacheSwapHashmapTask(std::vector<Tensor *> &inputs_,
                              std::vector<Tensor *> &outputs_,
                              int64_t &batch_size_, int64_t &hashmap_length_) {
  auto start = std::chrono::high_resolution_clock::now();

  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR(
        "CacheSwapHashmapKernel::CacheSwapHashmapTask: input or output is "
        "empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  HashmapEntry<T> *hashmap =
      reinterpret_cast<HashmapEntry<T> *>(inputs_[0]->GetData());
  T *miss_emb_idx = reinterpret_cast<T *>(inputs_[1]->GetData());
  T step_ = *reinterpret_cast<T *>(inputs_[2]->GetData());

  T *swap_cache_idx = reinterpret_cast<T *>(outputs_[0]->GetData());
  T *old_emb_idx = reinterpret_cast<T *>(outputs_[1]->GetData());

  int total_tag_count = 0;
  int total_delete_count = 0;
  int count_size = 1;
  for (int64_t i = 0; i < batch_size_; ++i) {
    if (miss_emb_idx[i] < 0) {
      swap_cache_idx[i] = -1;
      old_emb_idx[i] = -1;
    } else {
      T emb_idx = miss_emb_idx[i];
      T entry = HashFunc(emb_idx, hashmap_length_);
      int tag_count = 1;
      T ori = entry;

      // insert. find a entry is empty
      while (!hashmap[entry].IsEmpty()) {
        entry = (entry + 1) % hashmap_length_;
        if (ori == entry) {
          KERNEL_LOG_WARN("CacheSwapHashmapTask can not find entry.");
          break;
        }
        tag_count++;
      }

      hashmap[entry].step = step_;
      hashmap[entry].key = emb_idx;
      hashmap[entry].tag = tag_count;

      // delete, find a entry is not empty or not using
      T tmp_entry = (entry + 1) % hashmap_length_;
      int delete_count = 1;
      while (hashmap[tmp_entry].IsEmpty() ||
             hashmap[tmp_entry].IsUsing(step_)) {
        tmp_entry = (tmp_entry + 1) % hashmap_length_;
        delete_count += 1;
      }

      swap_cache_idx[i] = hashmap[tmp_entry].value;
      old_emb_idx[i] = hashmap[tmp_entry].key;
      hashmap[entry].value = swap_cache_idx[i];
      hashmap[tmp_entry].SetEmpty();
      int compress_count = Compress(hashmap, hashmap_length_, tmp_entry);

      count_size += 1;
      total_tag_count += tag_count;
      total_delete_count += (delete_count + compress_count);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  KERNEL_LOG_INFO("CacheSwapHashmap insert count %f.",
                  1.0 * total_tag_count / count_size);
  KERNEL_LOG_INFO("CacheSwapHashmap delete count %f.",
                  1.0 * total_delete_count / count_size);
  KERNEL_LOG_INFO(
      "CacheSwapHashmap execute %fms.",
      std::chrono::duration<double, std::milli>(end - start).count());
  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapHashmapKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                       std::vector<Tensor *> &, int64_t &,
                                       int64_t &)>>
      calls;
  calls[DT_INT32] = CacheSwapHashmapTask<int32_t>;
  calls[DT_INT64] = CacheSwapHashmapTask<int64_t>;

  auto iter = calls.find(param_type_);

  if (iter == calls.end()) {
    KERNEL_LOG_ERROR(
        "CacheSwapHashmapKernel op don't support input tensor types: %s",
        typeid(param_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return iter->second(inputs_, outputs_, batch_size_, hashmap_length_);
}

uint32_t CacheSwapHashmapKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CacheSwapHashmapKernel::GetInputAndCheck start!");

  // get input Tensors
  const int num_input = 3;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "CacheSwapHashmapKernel::GetInputAndCheck: get input tensor[%d] "
          "failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int num_output = 2;
  for (int i = 0; i < num_output; ++i) {
    Tensor *tensor = ctx.Output(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "CacheSwapHashmapKernel::GetInputAndCheck: get output tensor[%d] "
          "failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  KERNEL_LOG_INFO("CacheSwapHashmapKernel::GetInputAndCheck success!");

  std::shared_ptr<TensorShape> hashmap_shape = ctx.Input(0)->GetTensorShape();
  std::shared_ptr<TensorShape> emb_idx_shape = ctx.Input(1)->GetTensorShape();

  if (hashmap_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
        "CacheSwapHashmapKernel::GetInputAndCheck: only support hashmap rank "
        "2, but got %d",
        hashmap_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  hashmap_length_ = hashmap_shape->GetDimSize(0);
  if (hashmap_shape->GetDimSize(1) != 4) {
    KERNEL_LOG_ERROR(
        "CacheSwapHashmapKernel::GetInputAndCheck: only support hashmap shape "
        "(n, 4), but got (%d, %d)",
        hashmap_length_, hashmap_shape->GetDimSize(1));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (int i = 0; i < emb_idx_shape->GetDims(); ++i) {
    batch_size_ *= emb_idx_shape->GetDimSize(i);
  }

  return KERNEL_STATUS_OK;
}

uint32_t CacheSwapHashmapKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CacheSwapHashmapKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CacheSwapHashmapKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("CacheSwapHashmapKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(CACHE_SWAP_HASHMAP, CacheSwapHashmapKernel);
}  // namespace aicpu
