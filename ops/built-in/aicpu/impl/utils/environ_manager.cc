/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "utils/environ_manager.h"
#include "status.h"

namespace aicpu {
constexpr auto kScalarTensorShapeDim = 1;
constexpr auto kScalarTensorShapeSize = 1;

int64_t EnvironMgr::Create() {
  std::unique_lock<std::mutex> lock(mutex);
  if (env_handles_count_ >= INT64_MAX) {
    KERNEL_LOG_ERROR(" The handles number:%d is out of range: ", env_handles_count_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t ret_handle = ++env_handles_count_;
  auto env = std::make_shared<Environ>(ret_handle);
  KERNEL_CHECK_NULLPTR(env, KERNEL_STATUS_PARAM_INVALID,
                       "env is null.");
  envs_[ret_handle] = env;

  return ret_handle;
}

EnvironPtr EnvironMgr::Get(int64_t handle) {
  std::unique_lock<std::mutex> lock(mutex);
  const auto &envIter = envs_.find(handle);
  if (envIter != envs_.end()) {
    auto &result = envIter->second;
    return result;
  }
  return nullptr;
}

void EnvironMgr::Clear() {
  std::unique_lock<std::mutex> lock(mutex);
  for (auto &env : envs_) {
    KERNEL_CHECK_NULLPTR_VOID(env.second, "env is null.")
    env.second->Clear();
  }
  envs_.clear();
}

bool EnvironMgr::IsScalarTensor(const std::vector<int64_t> &shape) const {
  if (shape.empty()) {
    KERNEL_LOG_DEBUG("The shape is empty.");
    return true;
  }

  if ((shape.size() == kScalarTensorShapeDim)
      && (shape[0] == kScalarTensorShapeSize)) {
    KERNEL_LOG_DEBUG("The tensor is scalar.");
    return true;
  }
  return false;
}

bool EnvironMgr::CheckEnvInput(const CpuKernelContext &ctx) const {
  AttrValue *value_type_ptr = ctx.GetAttr(kEnvValueTypeAttr);
  // Check the value type attr
  KERNEL_CHECK_NULLPTR(value_type_ptr, false, "Get attr:[%s] failed.", kEnvValueTypeAttr)
  int64_t value_type_attr = value_type_ptr->GetInt();
  if ((value_type_attr != kObjectTypeTensorType) && (value_type_attr != kObjectTypeEnvType)) {
    KERNEL_LOG_ERROR("The value type is not supported: [%d]", value_type_attr);
    return false;
  }

  // Check the input handle.
  Tensor *handle_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(handle_tensor, false, "Tensor shape is null.")
  if (!IsScalarTensor(handle_tensor->GetTensorShape()->GetDimSizes())) {
    KERNEL_LOG_ERROR("The input handle checks invalid.");
    return false;
  }

  // Check the input key
  Tensor *key_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(key_tensor, false, "Tensor shape is null.")
  if (!IsScalarTensor(key_tensor->GetTensorShape()->GetDimSizes())) {
    KERNEL_LOG_ERROR("The input key checks invalid.");
    return false;
  }

  // Check the input value
  Tensor *value_tensor = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(value_tensor, false, "Tensor shape is null.")
  if ((value_type_attr == kObjectTypeEnvType) && (!IsScalarTensor(value_tensor->GetTensorShape()->GetDimSizes()))) {
    KERNEL_LOG_ERROR("The input value checks invalid.");
    return false;
  }

  return true;
}
}  // namespace aicpu
