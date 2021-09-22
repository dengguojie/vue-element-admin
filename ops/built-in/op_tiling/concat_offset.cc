/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file concat_offset.cpp
 * \brief
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

bool ConcatOffsetTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                        utils::OpRunInfo& run_info) {
  using namespace ge;
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);
  auto input_desc = operator_info->MutableInputDesc(1);
  const GeShape& x_shape = input_desc->MutableShape();
  int64_t input_num = x_shape.GetDimNum() == 0 ? 1 : x_shape.GetDim(0);
  run_info.AddTilingData(input_num);
  OP_LOGI(op_type, "tiling data = %ld.", input_num);
  run_info.SetBlockDim(1);
  return true;
}  // namespace optiling

// register tiling interface of the ConcatOffset op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(ConcatOffset, ConcatOffsetTiling);
}  // namespace optiling
