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

#include "environ_get.h"
#include <memory>
#include <string>
#include "securec.h"
#include "utils/kernel_util.h"
#include "utils/environ_manager.h"

namespace {
const char *kEnvironGet = "EnvironGet";
}

namespace aicpu {
uint32_t EnvironGetCpuKernel::InitKernel(const CpuKernelContext &ctx) {
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(ctx)) {
    KERNEL_LOG_ERROR("The input checks invalid. ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  output_ = ctx.Output(0);
  input_default_value_ = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(output_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed.")
  value_type_attr_ = ctx.GetAttr(kEnvValueTypeAttr)->GetInt();

  auto value_type = output_->GetDataType();
  auto value_shape = output_->GetTensorShape()->GetDimSizes();
  auto default_value_type = input_default_value_->GetDataType();
  auto default_value_shape = input_default_value_->GetTensorShape()->GetDimSizes();
  if ((value_type != default_value_type) || (value_shape != default_value_shape)) {
    KERNEL_LOG_ERROR("The env value checks invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  value_size_ = GetSizeByDataType(value_type);
  for (auto &i : value_shape) {
    value_size_ *= i;
  }
  return KERNEL_STATUS_OK;
}

uint32_t EnvironGetCpuKernel::Compute(CpuKernelContext &ctx) {
  auto &env_mgr = EnvironMgr::GetInstance();
  if (InitKernel(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Init Kernel failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto *input_handle = static_cast<int64_t *>(ctx.Input(0)->GetData());
  auto *input_key = static_cast<int64_t *>(ctx.Input(1)->GetData());

  // Get handle and key
  int64_t handle = input_handle[0];
  int64_t key = input_key[0];

  // Get env and value by handle and key
  const auto &env = env_mgr.Get(handle);
  KERNEL_CHECK_NULLPTR(env, KERNEL_STATUS_PARAM_INVALID,
                       "Get env [%d] failed", handle)
  const auto &env_value = env->Get(key);
  // Default value
  auto *value = input_default_value_->GetData();
  auto value_size = input_default_value_->GetDataSize();
  auto value_type = value_type_attr_;
  if (env_value != nullptr) {
    value = env_value->addr_;
    value_size = env_value->size_;
    value_type = env_value->value_type_;
  }

  // Check the env value size and type. The value size may be aligned, so must be greater then value_size_.
  if ((value_size < value_size_) || (value_type != value_type_attr_)) {
    KERNEL_LOG_ERROR("The env value checks invalid, value_size:%d, value_type:%d, value_type_attr_:%d",
                     value_size_, value_type, value_type_attr_);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto ret = memcpy_s(output_->GetData(), value_size_, value, value_size_);
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size[%zu] from env map to output[0] failed.",
                     value_size_);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kEnvironGet, EnvironGetCpuKernel);
}  // namespace aicpu
