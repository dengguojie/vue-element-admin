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
#include "cpu_kernel_register.h"

#include <mutex>

#include "aicpu_context.h"
#include "cpu_kernel.h"
#include "log.h"
#include "status.h"

namespace {
#define TYPE_REGISTAR(type, fun) type##Registerar(type, fun)
// protect creatorMap_
std::mutex g_mutex;
}  // namespace

namespace aicpu {
/*
 * regist kernel.
 */
bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Registerar TYPE_REGISTAR(type, fun);
  return true;
}

/*
 * get instance.
 * @return CpuKernelRegister &: CpuKernelRegister instance
 */
CpuKernelRegister &CpuKernelRegister::Instance() {
  static CpuKernelRegister instance;
  return instance;
}

/*
 * get cpu kernel.
 * param opType: the op type of kernel
 * @return shared_ptr<CpuKernel>: cpu kernel ptr
 */
std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernel(
    const std::string &opType) {
  std::unique_lock<std::mutex> lock(g_mutex);
  auto iter = creatorMap_.find(opType);
  if (iter != creatorMap_.end()) {
    return iter->second();
  }
  KERNEL_LOG_WARN("The kernel[%s] is not registered.", opType.c_str());
  return std::shared_ptr<CpuKernel>(nullptr);
}

/*
 * get all cpu kernel registered op types.
 * @return std::vector<string>: all cpu kernel registered op type
 */
std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypes() const {
  std::vector<std::string> ret;
  std::unique_lock<std::mutex> lock(g_mutex);;
  for (auto iter = creatorMap_.begin(); iter != creatorMap_.end(); ++iter) {
    ret.push_back(iter->first);
  }

  return ret;
}

/*
 * run cpu kernel.
 * param ctx: context of kernel
 * @return uint32_t: 0->success other->failed
 */
uint32_t CpuKernelRegister::RunCpuKernel(CpuKernelContext &ctx) {
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("RunCpuKernel[%s] begin.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_OP_NAME, type) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_ERROR("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  if (aicpu::SetOpname != nullptr) {
    (void)aicpu::SetOpname(type);
  }

  uint32_t ret = kernel->Compute(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  KERNEL_LOG_INFO("RunCpuKernel[%s] success.", type.c_str());
  return KERNEL_STATUS_OK;
}

CpuKernelRegister::Registerar::Registerar(const std::string &type,
                                          const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Instance().Register(type, fun);
}

// register creator, this function will call in the constructor
void CpuKernelRegister::Register(const std::string &type,
                                 const KERNEL_CREATOR_FUN &fun) {
  std::unique_lock<std::mutex> lock(g_mutex);
  std::map<std::string, KERNEL_CREATOR_FUN>::iterator iter =
      creatorMap_.find(type);
  if (iter != creatorMap_.end()) {
    KERNEL_LOG_WARN("Register[%s] creator already exist",
                    type.c_str());
    return;
  }

  creatorMap_[type] = fun;
  KERNEL_LOG_DEBUG("Kernel[%s] register successfully", type.c_str());
}
}  // namespace aicpu
