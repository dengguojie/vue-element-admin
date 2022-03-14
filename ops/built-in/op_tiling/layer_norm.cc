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

const struct ops::AttrBase LAYERNORM_BEGIN_NORM_AXIS(0, "begin_norm_axis");

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
    if (compile_info.count("reduce_mean_cof_dtype") > 0) {
      GetCompileValue(compile_info, "reduce_mean_cof_dtype", compile_value.reduce_mean_cof_dtype);
      // change str to Ge DataType
      compile_value.reduce_mean_cof_ge_dtype = GetGeTypeFromStr(compile_value.reduce_mean_cof_dtype);
    }
    // add for unknown axis mode
    (void)GetCompileValue(compile_info, "unknown_mode", compile_value.is_unknown_mode, false);
    if (!compile_value.is_unknown_mode) {
      OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_reduce_axis", compile_value.ori_reduce_axis),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormParseFunc, get _ori_reduce_axis error"),
                      return false);
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

bool LayerNormUnknowAxisTiling(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                               utils::OpRunInfo &run_info) {
  OP_LOGI(op_type.c_str(), "LayerNormUnknowAxisTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_x_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_x_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input x desc return nullptr!"), return false);
  const GeShape &input_shape = input_x_desc->MutableShape();
  std::size_t input_shape_dim = input_shape.GetDimNum();
  // get attr for reduce axis
  int32_t reduce_attr = 0;
  ops::GetAttrValue(op_paras, LAYERNORM_BEGIN_NORM_AXIS, reduce_attr);
  OP_TILING_CHECK(reduce_attr < 0,
                  OP_LOGD(op_type.c_str(), "BEGIN_NORM_AXIS is < 0, will do += input_x_shape"),
                  reduce_attr += input_shape_dim);
  OP_TILING_CHECK(reduce_attr < 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BEGIN_NORM_AXIS is < 0, return failed"),
                  return false);
  OP_TILING_CHECK(reduce_attr >= static_cast<int32_t>(input_shape_dim),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "BEGIN_NORM_AXIS is > input dim size, return failed"),
                  return false);
  std::vector<int32_t> reduce_axis(input_shape_dim - reduce_attr, 0);
  for (int32_t i = 0; i < input_shape_dim - reduce_attr; i++) {
    reduce_axis[i] = reduce_attr + i;
  }
  std::vector<int64_t> input_shape_vec = input_shape.GetDims();
  std::vector<std::vector<int64_t>> shapes = {input_shape_vec};
  std::vector<std::vector<int32_t>> axes{reduce_axis};
  ge::DataType input_x_dtype = input_x_desc->GetDataType();

  // now the norm parttern doesn't need this, so the shapes and input_x_dtype in norm_info is reserved
  OpInfo norm_info(shapes, input_x_dtype, axes);
  OP_TILING_CHECK(!op_info.tiling_handler->DoTiling(op_paras, run_info, norm_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormUnknowAxisTiling, do DoTiling failed"),
                  return false);
  OP_TILING_CHECK(op_info.reduce_mean_cof_dtype.empty(),
                  OP_LOGI(op_type.c_str(), "need not do AddReducMeanCof, return true"),
                  return true);
  OP_TILING_CHECK(!AddReducMeanCof(input_shape, op_info.reduce_mean_cof_ge_dtype, reduce_axis, run_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormUnknowAxisTiling, do AddReducMeanCof failed"),
                  return false);
  OP_LOGI(op_type.c_str(), "LayerNormUnknowAxisTiling end.");
  return true;
}

bool LayerNormTiling(const string &op_type, const ge::Operator &op_paras, const layerNormOpInfo &op_info,
                     utils::OpRunInfo &run_info) {
  if (op_info.is_support_vexp_pattern) {
    // norm template tiling_stratery
    OP_LOGI(op_type.c_str(), "LayerNormNormalTiling running.");

    // change to unknow reduce mode
    if (op_info.is_unknown_mode) {
      return LayerNormUnknowAxisTiling(op_type, op_paras, op_info, run_info);
    }
    bool ret = op_info.tiling_handler->DoTiling(op_paras, run_info);
    if (!ret) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_info.tiling_handler->DoTiling(op_paras, run_info) request failed.");
      return false;
    }
    OP_TILING_CHECK(op_info.reduce_mean_cof_dtype.empty(),
                    OP_LOGI(op_type.c_str(), "need not do AddReducMeanCof, return true"),
                    return true);
    const auto &input_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(0)->GetShape();
    OP_TILING_CHECK(!AddReducMeanCof(input_shape, op_info.reduce_mean_cof_ge_dtype,
                                     op_info.ori_reduce_axis, run_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "LayerNormTiling, do AddReducMeanCof failed"),
                    return false);
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
