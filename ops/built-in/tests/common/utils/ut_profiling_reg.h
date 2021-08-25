/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef COMMON_UTILS_UT_PROFILING_REG_H_
#define COMMON_UTILS_UT_PROFILING_REG_H_
#include <iostream>
#include "graph/types.h"
#include "graph/tensor.h"
#include <chrono>
#include <op_log.h>

#define PROFILING_TEST(funtion, arguments, profiling_num, target_cost)                                      \
  int64_t test_time = profiling_num;                                                                        \
  int64_t target_cost_per_loop = target_cost;                                                               \
  std::chrono::time_point<std::chrono::steady_clock> profiling_start_ts, profiling_end_ts;                  \
  profiling_start_ts = std::chrono::steady_clock::now();                                                    \
  for (int64_t i; i < test_time; i++) {                                                                     \
    funtion arguments;                                                                                      \
  }                                                                                                         \
  profiling_end_ts = std::chrono::steady_clock::now();                                                      \
  int64_t profiling_cast =                                                                                  \
      std::chrono::duration_cast<std::chrono::microseconds>(profiling_end_ts - profiling_start_ts).count(); \
  EXPECT_EQ(profiling_cast <= target_cost_per_loop * test_time, true);

#endif  // COMMON_UTILS_UT_PROFILING_REG_H_
