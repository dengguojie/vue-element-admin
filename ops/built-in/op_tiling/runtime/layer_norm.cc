/* Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "error_log.h"
#include "layer_norm.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

namespace optiling {
const struct ops::AttrBase LAYERNORM_BEGIN_NORM_AXIS(0, "begin_norm_axis");

static bool LayerNormParseFunc(const std::string& node_name, const nlohmann::json& compile_info,
                               LayerNormOpInfo* compile_value) {
  OP_TILING_CHECK(!GetCompileValue(compile_info, "is_support_vexp_pattern", compile_value->is_support_vexp_pattern),
                  VECTOR_INNER_ERR_REPORT_TILIING(node_name, "LayerNormParseFunc, get is_support_vexp_pattern error"),
                  return false);
  if (compile_value->is_support_vexp_pattern) {
    // use norm tiling template
    compile_value->dsl_compile_info = ParseAutoTiling("LayerNorm", compile_info);
    OP_TILING_CHECK(compile_value->dsl_compile_info == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(node_name, "CreateAutoTilingHandler return nullptr"), return false);
    if (compile_info.count("reduce_mean_cof_dtype") > 0) {
      OP_TILING_CHECK(!GetCompileValue(compile_info, "reduce_mean_cof_dtype", compile_value->reduce_mean_cof_dtype),
                      VECTOR_INNER_ERR_REPORT_TILIING(node_name, "LayerNormParseFunc, get reduce_mean_cof_dtype error"),
                      return false);
      // change str to Ge DataType
      compile_value->reduce_mean_cof_ge_dtype = GetGeTypeFromStr(compile_value->reduce_mean_cof_dtype);
    }
    // add for unknown axis mode
    (void)GetCompileValue(compile_info, "unknown_mode", compile_value->is_unknown_mode, false);
    if (!compile_value->is_unknown_mode) {
      OP_TILING_CHECK(!GetCompileValue(compile_info, "_ori_reduce_axis", compile_value->ori_reduce_axis),
                      VECTOR_INNER_ERR_REPORT_TILIING(node_name, "LayerNormParseFunc, get _ori_reduce_axis error"),
                      return false);
    }

    OP_LOGD(node_name, "GetCompileParams success.");
    return true;
  }

  VECTOR_INNER_ERR_REPORT_TILIING(node_name, "runtime tiling only support autotiling now.");
  return false;
}

ge::graphStatus TilingPrepareForLayerNorm(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingPrepareForLayerNorm.");
  LayerNormOpInfo* compile_info = MutableCompileInfo<LayerNormOpInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);

  OP_TILING_CHECK(!LayerNormParseFunc(context->GetNodeName(), *parsed_object_cinfo, compile_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "do TilingPrepareForLayerNorm failed"),
                  return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "end to do TilingPrepareForLayerNorm.");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LayerNormUnknowAxisTiling(gert::TilingContext* context, const LayerNormOpInfo* op_info) {
  OP_LOGD(context->GetNodeName(), "LayerNormUnknowAxisTiling running.");
  const gert::StorageShape* input_shape_cls = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_shape_cls);
  const gert::Tensor* input_tenser = context->GetInputTensor(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_tenser);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const gert::Shape& input_shape = input_shape_cls->GetStorageShape();
  const std::size_t input_shape_dim = input_shape.GetDimNum();

  // get attr for reduce axis
  const int64_t* begin_norm_axis = attrs->GetAttrPointer<int64_t>(LAYERNORM_BEGIN_NORM_AXIS.attr_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, begin_norm_axis);
  int32_t reduce_attr = *begin_norm_axis < 0
                            ? static_cast<int32_t>(*begin_norm_axis) + static_cast<int32_t>(input_shape_dim)
                            : static_cast<int32_t>(*begin_norm_axis);

  std::vector<int64_t> reduce_axis(input_shape_dim - reduce_attr, 0);
  for (int32_t i = 0; i < static_cast<int32_t>(input_shape_dim - reduce_attr); i++) {
    reduce_axis[i] = reduce_attr + i;
  }
  // do autotiling
  const std::vector<gert::Shape> input_gert_shapes = {input_shape};
  ge::DataType input_x_dtype = input_tenser->GetDataType();
  OpInfo norm_info(op_info->dsl_compile_info.get());
  norm_info.SetInputShape(&input_gert_shapes);
  norm_info.SetInputType(&input_x_dtype);
  norm_info.SetAxes(&reduce_axis);
  OP_TILING_CHECK(!DoAutoTiling(context, &norm_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoTiling failed"),
                  return ge::GRAPH_FAILED);

  // update mean cof
  gert::TilingData* tiling_data = context->GetRawTilingData();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);
  OP_TILING_CHECK(op_info->reduce_mean_cof_dtype.empty(),
                  OP_LOGD(context->GetNodeName(), "LayerNormUnknowAxisTiling end"), return ge::GRAPH_SUCCESS);

  OP_LOGD(context->GetNodeName(), "LayerNormUnknowAxisTiling will do AddReduceMeanCof");
  OP_TILING_CHECK(!AddReduceMeanCof(input_shape, op_info->reduce_mean_cof_ge_dtype, reduce_axis, tiling_data),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "do AddReduceMeanCof failed"),
                  return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(), "LayerNormUnknowAxisTiling end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLayerNorm(gert::TilingContext* context) {
  // compile info
  const LayerNormOpInfo* compile_info = reinterpret_cast<const LayerNormOpInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  if (compile_info->is_support_vexp_pattern) {
    // norm template tiling_stratery
    OP_LOGD(context->GetNodeName(), "LayerNormNormalTiling running.");

    // change to unknow reduce mode
    if (compile_info->is_unknown_mode) {
      return LayerNormUnknowAxisTiling(context, compile_info);
    }

    // do common autotiling
    OpInfo norm_info(compile_info->dsl_compile_info.get());
    OP_TILING_CHECK(!DoAutoTiling(context, &norm_info),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "call DoTiling failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(compile_info->reduce_mean_cof_dtype.empty(),
                    OP_LOGD(context->GetNodeName(), "need not do AddReduceMeanCof, LayerNormTiling end."),
                    return ge::GRAPH_SUCCESS);

    const gert::StorageShape* input_shape_cls = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input_shape_cls);
    const gert::Shape& input_shape = input_shape_cls->GetStorageShape();
    gert::TilingData* tiling_data = context->GetRawTilingData();
    OPS_CHECK_NULL_WITH_CONTEXT(context, tiling_data);
    OP_TILING_CHECK(
        !AddReduceMeanCof(input_shape, compile_info->reduce_mean_cof_ge_dtype, compile_info->ori_reduce_axis,
                          tiling_data),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "LayerNormTiling, do AddReduceMeanCof failed"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "LayerNormTiling end.");
    return ge::GRAPH_SUCCESS;
  }

  // layernorm will do special tiling_stratery
  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "runtime tiling only support autotiling now.");
  return ge::GRAPH_FAILED;
}

// register tiling interface of LayerNorm op.
IMPL_OP(LayerNorm).Tiling(TilingForLayerNorm).TilingParse<LayerNormOpInfo>(TilingPrepareForLayerNorm);
}  // namespace optiling
