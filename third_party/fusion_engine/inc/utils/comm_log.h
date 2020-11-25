/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#ifndef FUSION_ENGINE_UTILS_COMMON_COMMON_LOG_H_
#define FUSION_ENGINE_UTILS_COMMON_COMMON_LOG_H_

#include<sys/syscall.h>
#include<unistd.h>
#include<cstdint>
#include<string>
#include"toolchain/slog.h"

#define UTIL_MODULE_NAME static_cast<int>(FE)

inline pid_t CommGetTid() {
    thread_local static pid_t tid = syscall(__NR_gettid);
    return tid;
}

#define D_CM_LOGD(MOD_NAME, fmt, ...)            \
    if (CheckLogLevel(MOD_NAME, DLOG_DEBUG) == 1) \
    dlog_debug(MOD_NAME, "%lu %s:" #fmt "\n", CommGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_CM_LOGI(MOD_NAME, fmt, ...)            \
    if (CheckLogLevel(MOD_NAME, DLOG_INFO) == 1) \
    dlog_info(MOD_NAME, "%lu %s:" #fmt "\n", CommGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_CM_LOGW(MOD_NAME, fmt, ...)            \
    if (CheckLogLevel(MOD_NAME, DLOG_WARN) == 1) \
    dlog_warn(MOD_NAME, "%lu %s:" #fmt "\n", CommGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_CM_LOGE(MOD_NAME, fmt, ...)            \
    if (CheckLogLevel(MOD_NAME, DLOG_DEBUG) == 1) \
    dlog_error(MOD_NAME, "%lu %s:" #fmt "\n", CommGetTid(), __FUNCTION__, ##__VA_ARGS__)
#define D_CM_LOGV(MOD_NAME, fmt, ...)            \
    dlog_event(MOD_NAME, "%lu %s:" #fmt "\n", CommGetTid(), __FUNCTION__, ##__VA_ARGS__)


#define CM_LOGD(...) D_CM_LOGD(UTIL_MODULE_NAME, __VA_ARGS__)
#define CM_LOGI(...) D_CM_LOGI(UTIL_MODULE_NAME, __VA_ARGS__)
#define CM_LOGW(...) D_CM_LOGW(UTIL_MODULE_NAME, __VA_ARGS__)
#define CM_LOGE(...) D_CM_LOGE(UTIL_MODULE_NAME, __VA_ARGS__)
#define CM_LOGV(...) D_CM_LOGV(UTIL_MODULE_NAME, __VA_ARGS__)

#define CM_CHECK(cond, log_func, return_expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            return_expr;                      \
        }                                     \
    } while (0)

#define CM_MAKE_SHAREF(exec_expr0, exec_expr1)  \ 
    do {                                        \
        try                                     \
        {                                       \
            exec_expr0                          \
        }                                       \
        catch(...)                              \
        {                                       \
            CM_LOGE("Make shared failed");      \
            exec_expr1;                         \
        }                                       \
    } while (0);
    
#define CM_CHECK_NOTNULL(val)                   \
    do {                                        \
       if ((val) == nullptr) {                  \
           CM_LOGE("Parameter[%s] must not be null.", #val);    \
           return fe::PARAM_INVALID                             \
       }                                                        \
    } while (0);
#endif // FUSION_ENGINE_UTILS_COMMON_COMMON_LOG_H_