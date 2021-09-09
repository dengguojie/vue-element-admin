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

/*!
 * \file vector_tiling_log.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_VECTOR_TILING_LOG_H_
#define OPS_BUILT_IN_OP_TILING_VECTOR_TILING_LOG_H_

#include "op_log.h"

namespace optiling {
#define V_OP_TILING_CHECK(cond, log_func, return_expr)   \
  do {                                                   \
    if (!(cond)) {                                       \
      log_func;                                          \
      return_expr;                                       \
    }                                                    \
  } while (0)

#define V_CHECK_EQ(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) == (y)), log_func, return_expr)
#define V_CHECK_NE(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) != (y)), log_func, return_expr)
#define V_CHECK_GT(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) > (y)), log_func, return_expr)
#define V_CHECK_GE(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) >= (y)), log_func, return_expr)
#define V_CHECK_LT(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) < (y)), log_func, return_expr)
#define V_CHECK_LE(x, y, log_func, return_expr) V_OP_TILING_CHECK(((x) <= (y)), log_func, return_expr)
} // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_VECTOR_TILING_LOG_H_
