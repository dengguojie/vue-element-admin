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

#include "reduce_tiling_v2.h"
#include "../fusion_pass/common/fp16_t.hpp"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
using namespace ge;

namespace optiling {
bool KLDivTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                 utils::OpRunInfo& run_info) {
  OP_LOGD(op_type, "Enter KLDivTiling");

  using namespace utils;
  utils::Reduce reduce(op_type, op_paras, op_info, run_info);
  bool ret = reduce.DoTiling();
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "DoTiling failed!");
    return false;
  }
  ret = ret && reduce.WriteTilingData();

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"),
                  return false);
  auto dim0 = input_desc->MutableShape().GetDim(0);
  if (dim0 == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The dim0 cannot be zero!");
    return false;
  }
  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string& reduce_mean_cof_dtype = op_info.at("reduce_mean_cof_dtype").get<std::string>();
    float reduce_mean_cof = 1.0;
    if (reduce_mean_cof_dtype == "float32") {
      reduce_mean_cof = reduce_mean_cof / dim0;
      run_info.AddTilingData(reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      reduce_mean_cof = reduce_mean_cof / dim0;
      fe::fp16_t reduce_mean_cof_fp16 = reduce_mean_cof;
      run_info.AddTilingData(reduce_mean_cof_fp16);
      run_info.AddTilingData((uint16_t)0);
    }
    OP_LOGD(op_type.c_str(), "reduce mean cof: %f", reduce_mean_cof);
  }
  OP_LOGD(op_type, "Exit KLDivTiling");
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(KLDiv, KLDivTiling);
}  // namespace optiling
