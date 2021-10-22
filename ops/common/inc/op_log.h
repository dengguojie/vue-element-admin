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

#include <string>
#include <type_traits>
#include "external/graph/operator.h"

#if !defined( __ANDROID__) && !defined(ANDROID)
#include "toolchain/slog.h"
#else
#include <utils/Log.h>
#endif

#define OPPROTO_SUBMOD_NAME "OP_PROTO"

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type
get_op_name(const T& name) {
  return name.c_str();
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type
get_op_name(T name) {
  return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type
get_op_name(T name) {
  return name;
}

template <typename T>
std::string TbeGetName(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

template <typename T>
std::string TbeGetOpType(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetOpType(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

#define CHECK_DIVISOR_ZERO(divisor) \
  if (divisor == 0) {\
    return;\
  }

#define CHECK_DIVISOR_ZERO_RET(divisor,ret) \
  if (divisor == 0) {\
    return ret;\
  }

#if !defined( __ANDROID__) && !defined(ANDROID)
#define OP_LOGI(opname, ...) D_OP_LOGI(get_op_name(opname), __VA_ARGS__)
#define OP_LOGW(opname, ...) D_OP_LOGW(get_op_name(opname), __VA_ARGS__)
#define OP_LOGE(opname, ...) D_OP_LOGE(get_op_name(opname), __VA_ARGS__)
#define OP_LOGD(opname, ...) D_OP_LOGD(get_op_name(opname), __VA_ARGS__)
#define OP_EVENT(opname, ...) D_OP_EVENT(get_op_name(opname), __VA_ARGS__)
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
#define OP_EVENT(opname, ...)
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
#define D_OP_EVENT(opname, fmt, ...) DlogSub(TBE, OPPROTO_SUBMOD_NAME, DLOG_EVENT, " %s:%d OpName:[%s] "#fmt, __FUNCTION__, __LINE__, opname, ##__VA_ARGS__)
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
#define D_OP_EVENT(opname, fmt, ...)
#define D_FUSION_PASS_LOGI(fmt, ...)
#define D_FUSION_PASS_LOGW(fmt, ...)
#define D_FUSION_PASS_LOGE(fmt, ...)
#define D_FUSION_PASS_LOGD(fmt, ...)
#endif

#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (condition) {                                                                                           \
      OP_LOGE(get_op_name(op_name), fmt, ##__VA_ARGS__);                                                       \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OP_LOGW_IF(condition, op_name, fmt, ...)                                                               \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (condition) {                                                                                           \
      OP_LOGE(get_op_name(op_name), fmt, ##__VA_ARGS__);                                                       \
    }                                                                                                          \
  } while (0)

#endif //GE_OP_LOG_H
