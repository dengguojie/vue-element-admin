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
#include "mem_cpy.h"
#include <atomic>
#include "securec.h"

#include "cpu_kernel_utils.h"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMemCpy = "MemCopy";
const uint32_t kMaxNum = 128;
}  // namespace
namespace aicpu {
uint32_t MemCpyCpuKernel::Compute(CpuKernelContext &ctx) {
  AttrValue *num_ = ctx.GetAttr("num");
  int64_t num = num_->GetInt();
  Tensor *release_flag = ctx.Input(0);
  Tensor *data_size = ctx.Input(1);
  Tensor *src_ptr = ctx.Input(2);
  Tensor *dst_ptr = ctx.Input(3);

  KERNEL_CHECK_NULLPTR(release_flag, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] release_flag nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(data_size, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] data_size nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(src_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] src_ptr nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(dst_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] dst_ptr nullptr", ctx.GetOpType().c_str())

  KERNEL_LOG_INFO("[MemCopy] num: [%lld] input num: [%lld] output num: [%lld]",
                  num, src_ptr->NumElements(), dst_ptr->NumElements());
  if (num > kMaxNum) {
    KERNEL_LOG_ERROR("[MemCopy] Attr [num] [%lld].error, must be <= [%u]", num,
                     kMaxNum);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((num != release_flag->NumElements()) ||
      (num != data_size->NumElements()) || (num != src_ptr->NumElements()) ||
      (num != dst_ptr->NumElements())) {
    KERNEL_LOG_ERROR(
        "[MemCopy] Attr [num] [%lld].error, not equal tensor "
        "elments[%lld][%lld][%lld][%lld]",
        num, release_flag->NumElements(), data_size->NumElements(),
        src_ptr->NumElements(), dst_ptr->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  uint64_t *release_flag_value =
      reinterpret_cast<uint64_t*>(release_flag->GetData());
  uint64_t *data_size_value = reinterpret_cast<uint64_t*>(data_size->GetData());

  uint64_t *src_ptr_value = reinterpret_cast<uint64_t*>(src_ptr->GetData());
  uint64_t *dst_ptr_value = reinterpret_cast<uint64_t*>(dst_ptr->GetData());

  KERNEL_LOG_INFO(
      "[MemCopy] release_flag_value: [%llu] data_size_value: [%llu] "
      "src_ptr_value: [%llu], dst_ptr_value: [%llu]",
      release_flag_value, data_size_value, src_ptr_value, dst_ptr_value);

  KERNEL_CHECK_NULLPTR(release_flag_value, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] release_flag nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(data_size_value, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] data_size nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(src_ptr_value, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] src_ptr nullptr", ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(dst_ptr_value, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] dst_ptr nullptr", ctx.GetOpType().c_str())

  std::atomic<bool> task_flag(true);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      uintptr_t input_addr = static_cast<uintptr_t>(src_ptr_value[i]);
      void *input_ptr = reinterpret_cast<void*>(input_addr);

      uintptr_t output_addr = static_cast<uintptr_t>(dst_ptr_value[i]);
      void *output_ptr = reinterpret_cast<void*>(output_addr);

      if (data_size_value[i] == 0 || input_ptr == nullptr ||
          output_ptr == nullptr) {
        KERNEL_LOG_EVENT(
            "[MemCopy]: index: [%d], data_size_value: [%llu], input_ptr: [%p] "
            "output_ptr: [%p]",
            i, data_size_value[i], input_ptr, output_ptr);
        task_flag.store(false);
        continue;
      }

      uint32_t ret =
          CpuKernelAllocatorUtils::CheckOutputDataPtr(src_ptr_value[i]);
      if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_EVENT("[MemCopy]:input src data ptr invaild.");
        task_flag.store(false);
        continue;
      }

      auto mem_ret = memcpy_s(output_ptr, data_size_value[i], input_ptr,
                              data_size_value[i]);
      if (mem_ret != EOK) {
        KERNEL_LOG_ERROR("[MemCopy]:Failed to memcpy output data ret [%d].",
                         mem_ret);
        task_flag.store(false);
      }

      if (release_flag_value[i]) {
        ret = CpuKernelAllocatorUtils::DeleteOutputDataPtr(src_ptr_value[i]);
        if (ret != KERNEL_STATUS_OK) {
          KERNEL_LOG_ERROR("[MemCopy]:input src data ptr invaild.");
          task_flag.store(false);
        }
      }
    }
  };

  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, num, 1, task);
  if ((ret != KERNEL_STATUS_OK) || (!task_flag.load())) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kMemCpy, MemCpyCpuKernel);
}  // namespace aicpu
