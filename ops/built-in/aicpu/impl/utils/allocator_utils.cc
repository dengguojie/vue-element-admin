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
#include "allocator_utils.h"
#include <unordered_set>
#include "securec.h"

#include "cce/fwk_adpt_struct.h"
#include "log.h"
#include "status.h"

namespace {
  std::unordered_set<uint64_t> g_allocated_ptr;
}

namespace aicpu {
uint32_t CpuKernelAllocatorUtils::UpdateOutputDataTensor(
    const std::vector<int64_t> &dims, DataType type, void *data_ptr,
    int64_t input_data_size, Tensor *&outputResultTensor) {
  KERNEL_CHECK_NULLPTR(outputResultTensor, KERNEL_STATUS_PARAM_INVALID,
                       "outputResultTensor nullptr");
  KERNEL_CHECK_NULLPTR(data_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "data_ptr nullptr");
  KERNEL_LOG_INFO("UpdateOutputDataTensor::START!!");

  if (dims.empty()) {
    KERNEL_LOG_ERROR("UpdateOutputDataTensor dims size == 0.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t num_elements = 1;
  int64_t dim_size = 0;
  for (size_t i = 0; i < dims.size(); i++) {
    dim_size = dims[i];
    KERNEL_CHECK_ASSIGN_64S_MULTI(num_elements, dim_size, num_elements,
                                  KERNEL_STATUS_PARAM_INVALID);
  }

  int64_t data_size = 0;
  int32_t element_size = GetSizeByDataType(type);
  KERNEL_CHECK_ASSIGN_64S_MULTI(num_elements, element_size, data_size,
                                KERNEL_STATUS_PARAM_INVALID);
  if (data_size < 0) {
    KERNEL_LOG_ERROR("UpdateOutputDataTensor data_size[%lld].", data_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  int64_t shape_buffer_size = 0;
  KERNEL_CHECK_ASSIGN_64S_MULTI(dims.size(), sizeof(int64_t), shape_buffer_size,
                                KERNEL_STATUS_PARAM_INVALID);

  void *output_shape_ptr = malloc(shape_buffer_size);
  KERNEL_CHECK_NULLPTR(output_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "malloc error, size[%llu]!", shape_buffer_size);
  
  int32_t ret = memcpy_s(output_shape_ptr, shape_buffer_size, dims.data(),
                         shape_buffer_size);
  if (ret != EOK) {
    free(output_shape_ptr);
    KERNEL_LOG_ERROR("memcpy error, size[%llu], ret[%d]!", shape_buffer_size, ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  aicpu::FWKAdapter::ResultSummary *result_summary =
      reinterpret_cast<aicpu::FWKAdapter::ResultSummary *>(
          outputResultTensor->GetData());
  if (data_size == 0) {
    result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(nullptr);
    result_summary->raw_data_size = 0;
    result_summary->shape_data_ptr =
        reinterpret_cast<uint64_t>(output_shape_ptr);
    result_summary->shape_data_size = shape_buffer_size;
    (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
    KERNEL_LOG_INFO("UpdateOutputDataTensor:: empty tensor END!!");
    return KERNEL_STATUS_OK;
  }
  void *output_data_ptr = malloc(data_size);
  if (output_data_ptr == nullptr) {
    KERNEL_LOG_ERROR("malloc error, size[%llu]!", data_size);
    free(output_shape_ptr);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (data_size > input_data_size) {
    free(output_data_ptr);
    free(output_shape_ptr);
    KERNEL_LOG_ERROR(
        "data_size[%ld] mast less than "
        "input_data_size[%ld]!",
        data_size, input_data_size);
    return KERNEL_STATUS_INNER_ERROR;
  }
  ret = memcpy_s(output_data_ptr, data_size, data_ptr, data_size);
  if (ret != EOK) {
    free(output_data_ptr);
    free(output_shape_ptr);
    KERNEL_LOG_ERROR("memcpy_s error, size[%llu], ret[%d]!", data_size, ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  result_summary->raw_data_size = data_size;
  result_summary->raw_data_ptr = reinterpret_cast<uint64_t>(output_data_ptr);
  result_summary->shape_data_size = shape_buffer_size;
  result_summary->shape_data_ptr = reinterpret_cast<uint64_t>(output_shape_ptr);

  (void)g_allocated_ptr.insert(result_summary->raw_data_ptr);
  (void)g_allocated_ptr.insert(result_summary->shape_data_ptr);
  KERNEL_LOG_INFO("UpdateOutputDataTensor :: END!!");

  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelAllocatorUtils::CheckOutputDataPtr(const uint64_t data_ptr) {
  auto find_data_ptr = g_allocated_ptr.find(data_ptr);
  if ((find_data_ptr == g_allocated_ptr.end())) {
    KERNEL_LOG_ERROR("CheckOutputDataPtrr invalid [%llu].",data_ptr);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t CpuKernelAllocatorUtils::DeleteOutputDataPtr(const uint64_t data_ptr) {
  auto find_data_ptr = g_allocated_ptr.find(data_ptr);
  if (find_data_ptr != g_allocated_ptr.end()) {
    free(reinterpret_cast<void*>(data_ptr));
    g_allocated_ptr.erase(find_data_ptr);
  }
  KERNEL_LOG_EVENT("DeleteOutputDataPtr invalid [%llu].",data_ptr);
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
