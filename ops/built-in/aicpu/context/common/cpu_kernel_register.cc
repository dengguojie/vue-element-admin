/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of register
 */

#include "cpu_kernel_register.h"

#include "aicpu_context.h"
#include "cpu_kernel.h"
#include "log.h"
#include "status.h"
#include <iostream>

namespace {
#define TYPE_REGISTAR(type, fun) type##Registerar(type, fun)
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
  auto iter = creatorMap_.find(opType);
  if (iter != creatorMap_.end()) {
    return iter->second();
  }
  KERNEL_LOG_WARN("The kernel:%s is not registered.", opType.c_str());
  return std::shared_ptr<CpuKernel>(nullptr);
}

/*
 * get all cpu kernel registered op types.
 * @return std::vector<string>: all cpu kernel registered op type
 */
std::vector<std::string> CpuKernelRegister::GetAllRegisteredOpTypes() const {
  std::vector<std::string> ret;
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
  std::cout << "RunCpuKernel begin." << std::endl;
  std::string type = ctx.GetOpType();
  KERNEL_LOG_INFO("RunCpuKernel:%s begin.", type.c_str());
  auto kernel = GetCpuKernel(type);
  if (kernel == nullptr) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (aicpu::SetThreadLocalCtx != nullptr) {
    if (aicpu::SetThreadLocalCtx(aicpu::CONTEXT_KEY_OP_NAME, type) !=
        aicpu::AICPU_ERROR_NONE) {
      KERNEL_LOG_INFO("Set kernel name[%s] to context failed.", type.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  return kernel->Compute(ctx);
}

CpuKernelRegister::Registerar::Registerar(const std::string &type,
                                          const KERNEL_CREATOR_FUN &fun) {
  CpuKernelRegister::Instance().Register(type, fun);
}

// register creator, this function will call in the constructor
void CpuKernelRegister::Register(const std::string &type,
                                 const KERNEL_CREATOR_FUN &fun) {
  std::map<std::string, KERNEL_CREATOR_FUN>::iterator iter =
      creatorMap_.find(type);
  if (iter != creatorMap_.end()) {
    KERNEL_LOG_WARN("CpuKernelRegister::Register: %s creator already exist",
                    type.c_str());
    return;
  }

  creatorMap_[type] = fun;
  KERNEL_LOG_DEBUG("kernel:%s register successfully", type.c_str());
}
}  // namespace aicpu
