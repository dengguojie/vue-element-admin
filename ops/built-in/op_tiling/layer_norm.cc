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
#include "layer_norm_v1.h"
#include "op_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {
bool LayerNormParseFunc(const std::string &op_type, const nlohmann::json &compile_info,
                        layerNormOpInfo &compile_value) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "is_support_vexp_pattern", compile_value.is_support_vexp_pattern),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get is_support_vexp_pattern error"),
                  return false);
  if (compile_value.is_support_vexp_pattern) {
    // use norm tiling template
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

  OP_TILING_CHECK(!GetCompileValue(compile_info, "input_format", compile_value.input_format),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get input_format error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get core_num error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "begin_norm_axis", compile_value.begin_norm_axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get begin_norm_axis error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "begin_params_axis", compile_value.begin_params_axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get begin_params_axis error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "is_tik_support", compile_value.is_tik_support),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get is_tik_support error"),
                  return false);
  auto tik_iter_num = compile_info.find("tik_mode");
  if (tik_iter_num != compile_info.end()) {
    GetCompileValue(compile_info, "tik_mode", compile_value.tik_mode);
  } else {
    compile_value.tik_mode = TSDYNAMIC;
  }
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ub_max_byte", compile_value.ub_max_byte),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get ub_max_byte error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "atomic_clean_diff_shape", compile_value.atomic_clean_diff_shape),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get atomic_clean_diff_shape error"),
                  return false);
  if (!compile_value.is_tik_support) {
    if (compile_info.count("reduce_mean_cof_dtype") > 0) {
      GetCompileValue(compile_info, "reduce_mean_cof_dtype", compile_value.reduce_mean_cof_dtype);
    }
    OP_TILING_CHECK(!GetCompileValue(compile_info, "is_support_vexp", compile_value.is_support_vexp),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get is_support_vexp error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "common_info", compile_value.common_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get common_info error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "pattern_info", compile_value.pattern_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get pattern_info error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "ub_info", compile_value.ub_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get ub_info error"), return false);

    OP_TILING_CHECK(!GetCompileValue(compile_info, "reduce_axis", compile_value.reduce_axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get reduce_axis error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "max_ub_size_normal_fp16", compile_value.max_ub_size_normal_fp16),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get max_ub_size_normal_fp16 error"),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(compile_info, "max_ub_size_normal_fp32", compile_value.max_ub_size_normal_fp32),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get max_ub_size_normal_fp32 error"),
                    return false);
    auto iter_num = compile_info.find("mode");
    if (iter_num != compile_info.end()) {
      GetCompileValue(compile_info, "mode", compile_value.mode);
    } else {
      compile_value.mode = TSDYNAMIC;
    }
  }
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

bool LayerNormTiling(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                     utils::OpRunInfo &run_info) {
  if (op_info.is_support_vexp_pattern) {
    // norm template tiling_stratery
    OP_LOGI(op_type.c_str(), "LayerNormNormalTiling running.");
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
  } else {
    // layernorm special tiling_stratery
    bool ret = LayerNormTilingV1(op_type, op_paras, op_info, run_info);
    return ret;
  }
}

// register tiling interface of LayerNorm op.
REGISTER_OP_TILING_V3_CUSTOM(LayerNorm, LayerNormTiling, LayerNormParseFunc, layerNormOpInfo);
}  // namespace optiling