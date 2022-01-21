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

#include "environ_set.h"
#include "environ_create.h"
#include "securec.h"
#include "utils/environ_manager.h"

namespace {
const char *kEnvironSet = "EnvironSet";
}

namespace aicpu {
uint32_t EnvironSetCpuKernel::InitKernel(const CpuKernelContext &ctx) {
  auto &env_mgr = EnvironMgr::GetInstance();
  if (!env_mgr.CheckEnvInput(ctx)) {
    KERNEL_LOG_ERROR("The input checks invalid. ");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Check the output.
  output_handle_ = ctx.Output(0);
  input_handle_ = ctx.Input(0);
  input_value_ = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(output_handle_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0] failed.")
  auto output_shape = output_handle_->GetTensorShape()->GetDimSizes();
  if (!env_mgr.IsScalarTensor(output_shape)) {
    KERNEL_LOG_ERROR("The output handle is not equal of input handle.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Get value type.
  value_type_attr_ = ctx.GetAttr(kEnvValueTypeAttr)->GetInt();

  // Get value size.
  auto value_shape = input_value_->GetTensorShape()->GetDimSizes();
  value_size_ = GetSizeByDataType(input_value_->GetDataType());
  for (auto &i : value_shape) {
    value_size_ *= i;
  }
  return KERNEL_STATUS_OK;
}

uint32_t EnvironSetCpuKernel::Compute(CpuKernelContext &ctx) {
  if (InitKernel(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Init Kernel failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_key = ctx.Input(1);
  auto &env_mgr = EnvironMgr::GetInstance();

  auto *value_ptr = malloc(value_size_);
  KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID, "Malloc failed.")
  auto ret = memcpy_s(value_ptr, value_size_,
                      input_value_->GetData(), value_size_);
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size from input[2] to environ failed.",
                     value_size_);

  // Get handle and key.
  int64_t *handle_data = static_cast<int64_t *>(input_handle_->GetData());
  int64_t *key_data = static_cast<int64_t *>(input_key->GetData());

  // Set env member.
  const auto &env = env_mgr.Get(handle_data[0]);
  KERNEL_CHECK_NULLPTR(env, KERNEL_STATUS_PARAM_INVALID,
                       "Get handle[%d] failed.", handle_data[0]);

  auto env_value = std::make_shared<EnvironValue>(
      value_ptr, value_size_, value_type_attr_);
  env->Set(key_data[0], env_value);
  // Set output handle
  ret = memcpy_s(output_handle_->GetData(), sizeof(int64_t),
                 input_handle_->GetData(), sizeof(int64_t));
  KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size from input[0] to output[0] failed.",
                     value_size_);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kEnvironSet, EnvironSetCpuKernel);
}  // namespace aicpu