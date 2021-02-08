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

/*!
 * \file op_log.h
 * \brief
 */
#ifndef GE_OP_LOG_H
#define GE_OP_LOG_H

#if !defined( __ANDROID__) && !defined(ANDROID)
#include "toolchain/slog.h"
#else
#include <utils/Log.h>
#endif

#define OPPROTO_SUBMOD_NAME "OP_PROTO"

#if !defined( __ANDROID__) && !defined(ANDROID)
#define OP_LOGI(opname, ...) D_OP_LOGI(opname, __VA_ARGS__)
#define OP_LOGW(opname, ...) D_OP_LOGW(opname, __VA_ARGS__)
#define OP_LOGE(opname, ...) D_OP_LOGE(opname, __VA_ARGS__)
#define OP_LOGD(opname, ...) D_OP_LOGD(opname, __VA_ARGS__)
#define GE_OP_LOGI(opname, ...) GE_D_OP_LOGI(opname, __VA_ARGS__)
#define GE_OP_LOGW(opname, ...) GE_D_OP_LOGW(opname, __VA_ARGS__)
#define GE_OP_LOGE(opname, ...) GE_D_OP_LOGE(opname, __VA_ARGS__)
#define GE_OP_LOGD(opname, ...) GE_D_OP_LOGD(opname, __VA_ARGS__)
#define FUSION_PASS_LOGI(...) D_FUSION_PASS_LOGI(__VA_ARGS__)
#define FUSION_PASS_LOGW(...) D_FUSION_PASS_LOGW(__VA_ARGS__)
#define FUSION_PASS_LOGE(...) D_FUSION_PASS_LOGE(__VA_ARGS__)
#define FUSION_PASS_LOGD(...) D_FUSION_PASS_LOGD(__VA_ARGS__)
#else
#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)
#define OP_LOGE(opname, ...)
#define OP_LOGD(opname, ...)
#define FUSION_PASS_LOGI(...)
#define FUSION_PASS_LOGW(...)
#define FUSION_PASS_LOGE(...)
#define FUSION_PASS_LOGD(...)
#endif

#if !defined( __ANDROID__) && !defined(ANDROID)
#define D_OP_LOGI(opname, fmt, ...) DlogSub(TBE, OPPROTO_SUBMOD_NAME, DLOG_INFO, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) DlogSub(TBE, OPPROTO_SUBMOD_NAME, DLOG_WARN,  " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define D_OP_LOGE(opname, fmt, ...) DlogSub(TBE, OPPROTO_SUBMOD_NAME, DLOG_ERROR, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) DlogSub(TBE, OPPROTO_SUBMOD_NAME, DLOG_DEBUG, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define GE_D_OP_LOGI(opname, fmt, ...) DlogSub(GE, OPPROTO_SUBMOD_NAME, DLOG_INFO, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define GE_D_OP_LOGW(opname, fmt, ...) DlogSub(GE, OPPROTO_SUBMOD_NAME, DLOG_WARN,  " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define GE_D_OP_LOGE(opname, fmt, ...) DlogSub(GE, OPPROTO_SUBMOD_NAME, DLOG_ERROR, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define GE_D_OP_LOGD(opname, fmt, ...) DlogSub(GE, OPPROTO_SUBMOD_NAME, DLOG_DEBUG, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGI(fmt, ...) DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_INFO, " %s:%d "#fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGW(fmt, ...) DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_WARN,  " %s:%d "#fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGE(fmt, ...) DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_ERROR, " %s:%d "#fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define D_FUSION_PASS_LOGD(fmt, ...) DlogSub(FE, OPPROTO_SUBMOD_NAME, DLOG_DEBUG, " %s:%d "#fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define D_OP_LOGI(opname, fmt, ...)
#define D_OP_LOGW(opname, fmt, ...)
#define D_OP_LOGE(opname, fmt, ...)
#define D_OP_LOGD(opname, fmt, ...)
#define D_FUSION_PASS_LOGI(fmt, ...)
#define D_FUSION_PASS_LOGW(fmt, ...)
#define D_FUSION_PASS_LOGE(fmt, ...)
#define D_FUSION_PASS_LOGD(fmt, ...)
#endif

#define OP_CHECK(condition, log_func, do_expr)                                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (condition) {                                                                                           \
      log_func;                                                                                                \
      do_expr;                                                                                                 \
    }                                                                                                          \
  } while (0)

#endif //GE_OP_LOG_H
