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
#include "embedding_lookup.h"
#include <memory.h>
#include <atomic>
#include <securec.h>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kEmbeddingLookup = "EmbeddingLookup";
}

namespace aicpu {
uint32_t EmbeddingLookuptMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  switch (index_type_) {
    case DT_INT8: {
      EmbeddingLookuptMsCpuKernel::DoComputeForEachType<int8_t>(ctx);
      break;
    }
    case DT_INT16: {
      int16_t data_type = 0;
      EmbeddingLookuptMsCpuKernel::DoComputeForEachType<int16_t>(ctx);
      break;
    }
    case DT_INT32: {
      EmbeddingLookuptMsCpuKernel::DoComputeForEachType<int32_t>(ctx);
      break;
    }
    case DT_INT64: {
      EmbeddingLookuptMsCpuKernel::DoComputeForEachType<int64_t>(ctx);
      break;
    }
    default: {
      KERNEL_LOG_ERROR(
          "EmbeddingLookup only support int index tensor types, got [%s].",
          DTypeStr(index_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t EmbeddingLookuptMsCpuKernel::DoComputeForEachType(CpuKernelContext &ctx) {
  size_t type_size = GetSizeByDataType(param_type_);
  char *param = reinterpret_cast<char *>(ioAddrs_[0]);
  T *indices = reinterpret_cast<T *>(ioAddrs_[1]);
  char *emb = reinterpret_cast<char *>(ioAddrs_[3]);
  auto memset_ret =
      memset_s(emb, out_size_ * type_size, 0x00, out_size_ * type_size);
  if (memset_ret != EOK) {
    KERNEL_LOG_ERROR("Memset failed, result:[%d]", memset_ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  for (int64_t i = 0; i < i_size_; ++i) {
    indices[i] -= offset_;
  }
  int64_t cargo_size = val_size_ * type_size;
  int64_t emb_size = out_size_ * type_size;
  std::atomic<bool> task_flag(true);
  auto shardCopy = [&](int64_t start, int64_t end) {
    char *tmp = emb + start * cargo_size;
    for (int64_t i = start; i < end; ++i, tmp += cargo_size) {
      if (indices[i] < 0 || indices[i] >= axis_size_) {
        continue;
      }
      auto tmp_size = emb_size - i * cargo_size;
      int ret = memcpy_s(tmp, tmp_size, param + indices[i] * cargo_size,
                         cargo_size);
      if (ret != 0) {
        task_flag.store(false);
        KERNEL_LOG_ERROR("EmbeddingLookup memcpy failed, result [%d].", ret);
      }
    }
  };
  uint32_t ret_val = CpuKernelUtils::ParallelFor(ctx, i_size_, 1, shardCopy);
  if (ret_val != KERNEL_STATUS_OK || !task_flag.load()) {
    KERNEL_LOG_ERROR("ParallelFor failed");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t EmbeddingLookuptMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  // get input Tensors
  const int kNumInput = 3;
  for (int i = 0; i < kNumInput; ++i) {
    Tensor *tensor = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "Get input:[%d] failed", i);
    ioAddrs_.push_back(reinterpret_cast<void *>(tensor->GetData()));
  }
  // get output Tensors
  const int kNumOutput = 1;
  for (int i = 0; i < kNumOutput; ++i) {
    Tensor *tensor = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "Get output:[%d] failed", i);
    ioAddrs_.push_back(reinterpret_cast<void *>(tensor->GetData()));
  }

  Tensor *param_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(param_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[0] failed.");
  std::shared_ptr<TensorShape> param_shape = param_tensor->GetTensorShape();

  Tensor *index_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(index_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[1] failed.");
  std::shared_ptr<TensorShape> index_shape = index_tensor->GetTensorShape();

  param_type_ = static_cast<DataType>(param_tensor->GetDataType());
  index_type_ = static_cast<DataType>(index_tensor->GetDataType());

  switch (index_type_) {
    case DT_INT8: {
      offset_ = *reinterpret_cast<int8_t *>(ioAddrs_[2]);
      break;
    }
    case DT_INT16: {
      offset_ = *reinterpret_cast<int16_t *>(ioAddrs_[2]);
      break;
    }
    case DT_INT32: {
      offset_ = *reinterpret_cast<int32_t *>(ioAddrs_[2]);
      break;
    }
    case DT_INT64: {
      offset_ = *reinterpret_cast<int64_t *>(ioAddrs_[2]);
      break;
    }
    default: {
      KERNEL_LOG_ERROR(
          "EmbeddingLookup only support int index tensor types, got [%s].",
          DTypeStr(index_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  p_size_ = param_shape->NumElements();
  i_size_ = index_shape->NumElements();

  for (int32_t i = 1; i < param_shape->GetDims(); ++i) {
    val_size_ *= param_shape->GetDimSize(i);
  }

  out_size_ *= i_size_ * val_size_;
  axis_size_ = param_shape->GetDimSize(0);
  KERNEL_CHECK_FALSE((ioAddrs_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "The size of input and output must be [4], but got [%zu].",
                     ioAddrs_.size());

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kEmbeddingLookup, EmbeddingLookuptMsCpuKernel);
}  // namespace aicpu
