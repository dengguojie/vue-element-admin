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
#ifndef AICPU_FOLDING_LOG_
#define AICPU_FOLDING_LOG_

#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "toolchain/slog.h"

#define GET_TID() syscall(__NR_gettid)
const char CPU_MODULE[] = "AICPU";

#ifdef RUN_TEST
#define CPU_LOG_DEBUG(fmt, ...)                                              \
  printf("[DEBUG] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", CPU_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_INFO(fmt, ...)                                              \
  printf("[INFO] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", CPU_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_WARN(fmt, ...)                                              \
  printf("[WARN] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", CPU_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_ERROR(fmt, ...)                                              \
  printf("[ERROR] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", CPU_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_EVENT(fmt, ...)                                              \
  printf("[EVENT] [%s][%s][%s:%d][tid:%ld]:" fmt "\n", CPU_MODULE, __FILE__, \
         __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#else
#define AICPU_MODULE_NAME static_cast<int32_t>(AICPU)
#define CPU_LOG_DEBUG(fmt, ...)                                          \
  dlog_debug(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%ld]:" fmt, CPU_MODULE, \
             __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_INFO(fmt, ...)                                          \
  dlog_info(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%ld]:" fmt, CPU_MODULE, \
            __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_WARN(fmt, ...)                                          \
  dlog_warn(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%ld]:" fmt, CPU_MODULE, \
            __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_ERROR(fmt, ...)                                          \
  dlog_error(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%ld]:" fmt, CPU_MODULE, \
             __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define CPU_LOG_EVENT(fmt, ...)                                          \
  dlog_event(AICPU_MODULE_NAME, "[%s][%s:%d][tid:%ld]:" fmt, CPU_MODULE, \
             __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#endif

#define CPU_CHECK_NULLPTR_VOID(value, logText...) \
  if (value == nullptr) {                         \
    CPU_LOG_ERROR(logText);                       \
    return;                                       \
  }

#define CPU_CHECK_NULLPTR(value, errorCode, logText...) \
  if (value == nullptr) {                               \
    CPU_LOG_ERROR(logText);                             \
    return errorCode;                                   \
  }

#define CPU_CHECK_NULLPTR_WARN(value, errorCode, logText...) \
  if (value == nullptr) {                                    \
    CPU_LOG_WARN(logText);                                   \
    return errorCode;                                        \
  }

#endif  // AICPU_FOLDING_LOG_
