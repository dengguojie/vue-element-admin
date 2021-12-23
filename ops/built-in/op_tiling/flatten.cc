/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file flatten.cc
 * \brief
 */
#include <math.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"

namespace optiling {
using namespace ge;
constexpr int32_t index_two = 2;

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "block_size"};

static int64_t GetCeilInt(int64_t value1, int64_t value2) {
  OP_TILING_CHECK(value2 == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("Flattern", "In the GetCeilInt function, the divisor is 0"),
                  return value1);
  return static_cast<int64_t>((value1 + value2 - 1) / value2);
}

bool GetFlattenCompileParams(const std::string& op_type, const std::vector<int64_t>& op_compile_info, int64_t& core_num,
                             int64_t& ub_size, int64_t& block_size) {
  OP_TILING_CHECK(
      op_compile_info.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), op_compile_info.size()),
      return false);

  core_num = op_compile_info[0];
  ub_size = op_compile_info[1];
  block_size = op_compile_info[index_two];

  return true;
}

bool FlattenTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_info,
                   utils::OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "FlattenTiling running.");
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);

  auto src_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(src_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get InputDesc failed."), return false);

  auto dst_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(dst_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OutputDesc failed."),
                  return false);

  const GeShape& src_shape = src_desc->MutableShape();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int64_t data_size;
  if (src_shape.GetDimNum() == 0) {
    data_size = 1;
  } else {
    data_size = src_shape.GetShapeSize();
  }

  int64_t core_num = 0;
  int64_t ub_size = 0;
  int64_t block_size = 0;
  if (!GetFlattenCompileParams(op_type, op_info, core_num, ub_size, block_size)) {
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  int64_t core_number = core_num;
  if (data_size < block_size) {
    core_number = 1;
  }
  int64_t core_data = GetCeilInt(data_size, core_number);
  core_data = GetCeilInt(core_data, block_size) * block_size;
  int64_t core_used = GetCeilInt(data_size, core_data);
  int64_t core_last = core_data;
  if (data_size % core_data != 0) {
    core_last = data_size % core_data;
  }

  int64_t copy_loop = core_data / ub_size;
  int64_t copy_tail = core_data % ub_size;
  int64_t last_copy_loop = core_last / ub_size;
  int64_t last_copy_tail = core_last % ub_size;

  OP_LOGD(op_type.c_str(),
          "CompileParams, core_data = %d, core_used = %d, copy_loop = %d, copy_tail = %d, last_copy_loop = %d, "
          "last_copy_tail = %d",
          core_data, core_used, copy_loop, copy_tail, last_copy_loop, last_copy_tail);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  run_info.AddTilingData(core_data);
  run_info.AddTilingData(core_used);
  run_info.AddTilingData(copy_loop);
  run_info.AddTilingData(copy_tail);
  run_info.AddTilingData(last_copy_loop);
  run_info.AddTilingData(last_copy_tail);

  run_info.SetBlockDim(core_used);
  PROFILING_TILING_END();
  OP_LOGI(op_type.c_str(), "FlattenTiling run success.");
  return true;
}

// register tiling interface of the Flatten op.
REGISTER_OP_TILING_V3_WITH_VECTOR(Flatten, FlattenTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
