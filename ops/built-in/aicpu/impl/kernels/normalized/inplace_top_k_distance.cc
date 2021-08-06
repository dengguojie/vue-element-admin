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
#include "inplace_top_k_distance.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kInplaceTopKDistance = "InplaceTopKDistance";
}

namespace aicpu {
uint32_t InplaceTopKDistanceCpuKernel::Compute(CpuKernelContext &ctx) {
  Inputs inputs;
  uint32_t res = GetInputAndCheck(ctx, inputs);
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("InplaceTopKDistance kernel GetInputAndCheck failed, KernelStatus is [%d]", res);
    return res;
  }

  DataType data_type = inputs.topk_pq_distance->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      KERNEL_LOG_INFO("InplaceTopKDistance Compute DT_FLOAT16");
      res = DoCompute<Eigen::half>(ctx, inputs);
      break;
    case DT_FLOAT:
      KERNEL_LOG_INFO("InplaceTopKDistance compute DT_FLOAT");
      res = DoCompute<float>(ctx, inputs);
      break;
    default:
      KERNEL_LOG_INFO(
          "InplaceTopKDistance input topk_pq_distance only support type[DT_FLOAT16, DT_FLOAT], but got type[%s]",
          DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_INFO("InplaceTopKDistance kernel compute failed, KernelStatus is [%d]", res);
    return res;
  }
  return KERNEL_STATUS_OK;
}
uint32_t InplaceTopKDistanceCpuKernel::GetInputAndCheck(CpuKernelContext& ctx, Inputs& inputs) {
  KERNEL_LOG_INFO("InplaceTopKDistance getInputAndCheck begin");
  inputs.topk_pq_distance = ctx.Input(0);
  inputs.topk_pq_index = ctx.Input(1);
  inputs.topk_pq_ivf = ctx.Input(2);
  inputs.pq_distance = ctx.Input(3);
  inputs.pq_index = ctx.Input(4);
  inputs.pq_ivf = ctx.Input(5);

  KERNEL_CHECK_NULLPTR(inputs.topk_pq_distance, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0], name[topk_pq_distance] failed");
  KERNEL_CHECK_NULLPTR(inputs.topk_pq_index, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[topk_pq_index] failed");
  KERNEL_CHECK_NULLPTR(inputs.topk_pq_ivf, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[topk_pq_ivf] failed");
  KERNEL_CHECK_NULLPTR(inputs.pq_distance, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[pq_distance] failed");
  KERNEL_CHECK_NULLPTR(inputs.pq_index, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[pq_index] failed");
  KERNEL_CHECK_NULLPTR(inputs.pq_ivf, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[pq_ivf] failed");

  KERNEL_LOG_INFO("InplaceTopKDistance getInputAndCheck topk_pq_distance NumElements is[%d]",
                  inputs.topk_pq_distance->NumElements());
  KERNEL_LOG_INFO("InplaceTopKDistance getInputAndCheck topk_pq_distance shape is[%d]",
                  inputs.topk_pq_distance->GetTensorShape()->GetDims());
  KERNEL_LOG_INFO("InplaceTopKDistance getInputAndCheck pq_ivf shape is[%d]",
                  inputs.pq_ivf->GetTensorShape()->GetDims());

  inputs.order = ctx.GetAttr("order");
  KERNEL_CHECK_NULLPTR(inputs.order, KERNEL_STATUS_PARAM_INVALID, "Get attr, name[order] failed");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InplaceTopKDistanceCpuKernel::DoCompute(CpuKernelContext& ctx, Inputs& inputs) {
  uint64_t topk_elements_num = inputs.topk_pq_distance->NumElements();
  std::vector<Item<T>> items;
  T *topk_value_ptr = static_cast<T *>(inputs.topk_pq_distance->GetData());
  int32_t *topk_index_ptr = static_cast<int32_t *>(inputs.topk_pq_index->GetData());
  int32_t *topk_ivf_ptr = static_cast<int32_t *>(inputs.topk_pq_ivf->GetData());
  for (uint64_t i = 0; i < topk_elements_num; i++) {
    T topk_value = topk_value_ptr[i];
    int32_t topk_index = topk_index_ptr[i];
    int32_t topk_ivf = topk_ivf_ptr[i];
    items.push_back({topk_value, topk_index, topk_ivf});
  }

  uint64_t elements_num = inputs.pq_distance->NumElements();
  T *new_value_ptr = static_cast<T *>(inputs.pq_distance->GetData());
  int32_t *new_index_ptr = static_cast<int32_t *>(inputs.pq_index->GetData());
  int32_t *new_ivf_ptr = static_cast<int32_t *>(inputs.pq_ivf->GetData());
  for (uint64_t i = 0; i < elements_num; i++) {
    T new_value = new_value_ptr[i];
    int32_t new_index = new_index_ptr[i];
    int32_t new_ivf = *new_ivf_ptr;
    items.push_back({new_value, new_index, new_ivf});
  }

  sort(items.begin(), items.end(), [](const Item<T>& a, const Item<T>& b) { return a.value < b.value; });
  return ModifyInput(items, inputs, ctx);
}

template <typename T>
uint32_t InplaceTopKDistanceCpuKernel::ModifyInput(std::vector<Item<T>> items_vec, Inputs& inputs,
                                                   CpuKernelContext& ctx) {
  uint64_t topk_elements_num = inputs.topk_pq_distance->NumElements();
  T *topk_value_ptr = static_cast<T *>(inputs.topk_pq_distance->GetData());
  int32_t *topk_index_ptr = static_cast<int32_t *>(inputs.topk_pq_index->GetData());
  int32_t *topk_ivf_ptr = static_cast<int32_t *>(inputs.topk_pq_ivf->GetData());

  std::string order = inputs.order->GetString();
  KERNEL_LOG_INFO("InplaceTopKDistance attr order is [%s]", order.c_str());

  if ("asc" == order) {
    KERNEL_LOG_INFO("InplaceTopKDistance modifyInput asc begin");
    uint64_t j = 0;
    for (auto i = items_vec.begin(); i < items_vec.begin() + topk_elements_num; i++, j++) {
      topk_value_ptr[j] = (*i).value;
      topk_index_ptr[j] = (*i).index;
      topk_ivf_ptr[j] = (*i).ivf;
    }
  } else {
    KERNEL_LOG_INFO("InplaceTopKDistance modifyInput desc begin");
    uint64_t j = topk_elements_num - 1;
    for (auto i = items_vec.end() - topk_elements_num; i < items_vec.end(); i++, j--) {
      topk_value_ptr[j] = (*i).value;
      topk_index_ptr[j] = (*i).index;
      topk_ivf_ptr[j] = (*i).ivf;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kInplaceTopKDistance, InplaceTopKDistanceCpuKernel);
}  // namespace aicpu