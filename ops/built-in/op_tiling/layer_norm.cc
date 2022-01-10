/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "../fusion_pass/common/fp16_t.hpp"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {

struct layerNormOpInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  string reduce_mean_cof_dtype;
  std::vector<int32_t> ori_reduce_axis;
};

bool LayerNormParseFunc(const std::string &op_type, const nlohmann::json &compile_info,
                        layerNormOpInfo &compile_value) {
  compile_value.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_NORM, compile_info);
  OP_TILING_CHECK(compile_value.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_reduce_axis", compile_value.ori_reduce_axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get _ori_reduce_axis error"),
                  return false);
  if (compile_info.count("reduce_mean_cof_dtype") > 0) {
    GetCompileValue(compile_info, "reduce_mean_cof_dtype", compile_value.reduce_mean_cof_dtype);
  }
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool LayerNormTiling(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                     utils::OpRunInfo &run_info) {
  OP_LOGI(op_type.c_str(), "LayerNormTiling running.");
  bool ret = op_info.tiling_handler->DoTiling(op_paras, run_info);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info.tiling_handler->DoTiling(op_paras, run_info) request failed.");
    return false;
  }
  if (op_info.reduce_mean_cof_dtype.empty()) {
    return ret;
  }
  const auto &input_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(0)->GetShape();
  std::size_t dim_len = input_shape.GetDimNum();
  std::size_t ori_reduce_axis_len = op_info.ori_reduce_axis.size();
  float reduce_mean_cof = 1.0;
  for (std::size_t i = 0; i < ori_reduce_axis_len; i++) {
    int32_t single_reduce_axis = op_info.ori_reduce_axis[i];
    // convert reduce axis (-1 -> dim_len-1)
    if (single_reduce_axis < 0) {
      single_reduce_axis = dim_len + single_reduce_axis;
    }
    // check reduce axis value
    V_OP_TILING_CHECK(
        (single_reduce_axis < static_cast<int32_t>(dim_len)),
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "value of reduce axis %d is illegel", single_reduce_axis),
        return false);
    reduce_mean_cof = reduce_mean_cof / input_shape.GetDim(single_reduce_axis);
  }
  const string &reduce_mean_cof_dtype = op_info.reduce_mean_cof_dtype;
  if (reduce_mean_cof_dtype == "float32") {
    run_info.AddTilingData((float)reduce_mean_cof);
  } else if (reduce_mean_cof_dtype == "float16") {
    run_info.AddTilingData((fe::fp16_t)reduce_mean_cof);
    run_info.AddTilingData((uint16_t)0);
  }
  OP_LOGI(op_type.c_str(), "LayerNormTiling end.");
  return ret;
}

// register tiling interface of LayerNorm op.
REGISTER_OP_TILING_V3_CUSTOM(LayerNorm, LayerNormTiling, LayerNormParseFunc, layerNormOpInfo);
}  // namespace optiling