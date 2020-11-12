/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: log
 */

#ifndef CPU_KERNELS_LOG_H
#define CPU_KERNELS_LOG_H

#include <unistd.h>
#include <stdio.h>
#include <sys/syscall.h>
#include "toolchain/slog.h"

#define GET_TID() syscall(__NR_gettid)
const char KERNEL_MODULE[] = "AICPU";

#ifdef RUN_TEST
#define KERNEL_LOG_DEBUG(fmt, ...)                                                                                   \
    printf("[DEBUG] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, __FUNCTION__, __LINE__, GET_TID(), \
        ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...)                                                                                   \
    printf("[INFO] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, __FUNCTION__, __LINE__, GET_TID(), \
        ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...)                                                                                   \
    printf("[WARN] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, __FUNCTION__, __LINE__, GET_TID(), \
        ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...)                                                                                   \
    printf("[ERROR] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, __FUNCTION__, __LINE__, GET_TID(), \
        ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...)                                                                                   \
    printf("[EVENT] [%s][%s][%s:%d][tid:%lu]:" fmt "\n", KERNEL_MODULE, __FILE__, __FUNCTION__, __LINE__, GET_TID(), \
        ##__VA_ARGS__)
#else
#define KERNEL_LOG_DEBUG(fmt, ...) \
    dlog_debug(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_INFO(fmt, ...) \
    dlog_info(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_WARN(fmt, ...) \
    dlog_warn(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_ERROR(fmt, ...) \
    dlog_error(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#define KERNEL_LOG_EVENT(fmt, ...) \
    dlog_event(AICPU, "[%s][%s:%d][tid:%lu]:" fmt, KERNEL_MODULE, __FUNCTION__, __LINE__, GET_TID(), ##__VA_ARGS__)
#endif

#define KERNEL_CHECK_NULLPTR_VOID(value, logText...) if (value == nullptr) {                          \
        KERNEL_LOG_ERROR(logText);                   \
        return;                                      \
    }

#define KERNEL_CHECK_NULLPTR(value, errorCode, logText...) if (value == nullptr) {                                \
        KERNEL_LOG_ERROR(logText);                         \
        return errorCode;                                  \
    }

#define KERNEL_CHECK_ASSIGN_64S_MULTI(A, B, result, errorCode)                      \
    if ((A) != 0 && (B) != 0 && ((INT64_MAX) / (A)) <= (B)) {                       \
        KERNEL_LOG_ERROR("Integer reversed multiA: %llu * multiB: %llu", (A), (B)); \
        return errorCode;                                                           \
    }                                                                               \
    (result) = ((A) * (B));

#endif // CPU_KERNELS_LOG_H
