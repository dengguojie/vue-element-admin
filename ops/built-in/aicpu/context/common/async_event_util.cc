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

#include "async_event_util.h"
#include <dlfcn.h>
#include "log.h"

namespace {
const char *kSharderPath = "/usr/lib64/libaicpu_sharder.so";
const char *kNotifyWaitFunc = "AicpuNotifyWait";
const char *kRegEventCbFunc = "AicpuRegEventCb";
}  // namespace

namespace aicpu {

AsyncEventUtil &AsyncEventUtil::GetInstance() {
    static AsyncEventUtil async_event_util;
    return async_event_util;
}

AsyncEventUtil::AsyncEventUtil() {
  sharder_ = dlopen(kSharderPath, RTLD_LAZY | RTLD_GLOBAL);
  if (sharder_ == nullptr) {
    KERNEL_LOG_WARN("Device sharder dlopen so [%s] failed, error[%s]",
                    kSharderPath, dlerror());
    notify_wait_func_ = nullptr;
    reg_event_cb_func_ = nullptr;
  } else {
    notify_wait_func_ = reinterpret_cast<NotifyWaitFunc>(dlsym(sharder_, kNotifyWaitFunc));
    if (notify_wait_func_ == nullptr) {
      KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kNotifyWaitFunc, dlerror());
    }
    reg_event_cb_func_ = reinterpret_cast<RegEventCbFunc>(dlsym(sharder_, kRegEventCbFunc));
    if (reg_event_cb_func_ == nullptr) {
      KERNEL_LOG_WARN("Get Function[%s] address failed, error[%s]", kRegEventCbFunc, dlerror());
    }

    KERNEL_LOG_INFO("Device sharder dlopen so[%s] success.", kSharderPath);
  }
}

AsyncEventUtil::~AsyncEventUtil() {
  if (sharder_ != nullptr) {
    (void)dlclose(sharder_);
  }
}

void AsyncEventUtil::NotifyWait(void *notify_param, const uint32_t param_len) {
  if (notify_wait_func_ != nullptr) {
    notify_wait_func_(notify_param, param_len);
    return;
  }
  KERNEL_LOG_WARN("Function[%s] is null", kNotifyWaitFunc);
}

bool AsyncEventUtil::RegEventCb(const uint32_t event_id, const uint32_t sub_event_id,
                                const std::function<void(void *)> &cb) {
  if (reg_event_cb_func_ != nullptr) {
    return reg_event_cb_func_(event_id, sub_event_id, cb);
  }
  KERNEL_LOG_WARN("Function[%s] is null.", kRegEventCbFunc);
  return false;
}

}  // namespace aicpu

