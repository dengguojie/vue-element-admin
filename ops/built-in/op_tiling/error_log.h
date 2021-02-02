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
 * \file error_log.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

namespace optiling {

#define CHECK(cond, message, ...)   \
  if (!cond) {                      \
    OP_LOGE(#message, __VA_ARGS__); \
    return false;                   \
  };

#define CHECK_EQ(x, y, message, ...) CHECK(((x) == (y)), message, __VA_ARGS__);
#define CHECK_NE(x, y, message, ...) CHECK(((x) != (y)), message, __VA_ARGS__);
#define CHECK_GT(x, y, message, ...) CHECK(((x) > (y)), message, __VA_ARGS__);
#define CHECK_GE(x, y, message, ...) CHECK(((x) >= (y)), message, __VA_ARGS__);
#define CHECK_LT(x, y, message, ...) CHECK(((x) < (y)), message, __VA_ARGS__);
#define CHECK_LE(x, y, message, ...) CHECK(((x) <= (y)), message, __VA_ARGS__);

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
