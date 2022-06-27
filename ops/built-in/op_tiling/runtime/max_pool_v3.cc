/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "max_pool_v3.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"
#include "op_util.h"

using namespace ge;

namespace optiling {
constexpr size_t INDEX_0 = 0;
constexpr size_t INDEX_1 = 1;
constexpr size_t INDEX_2 = 2;
constexpr size_t INDEX_3 = 3;
constexpr size_t INDEX_4 = 4;
constexpr size_t DIMNUM_NC1HWC0 = 5;
const int64_t DIMSIZE_C0 = 16;
constexpr int32_t PADDING_VALUE = 2;
constexpr int32_t TILING_MODE_0 = 0;
constexpr int32_t TILING_MODE_1 = 1;
constexpr int32_t TILING_MODE_2 = 2;
constexpr int32_t TILING_MODE_3 = 3;
constexpr int32_t TILING_MODE_4 = 4;
constexpr int32_t TILING_MODE_5 = 5;
constexpr int32_t TILING_MODE_6 = 6;
constexpr int32_t TILING_MODE_7 = 7;
constexpr size_t TWICE = 2;

static void CalCoreNum(MaxPoolV3TilingData* param, int32_t total_ele, int32_t core_num) {
  param->one_core_ele = CeilDiv(total_ele, core_num);
  param->act_core_num = total_ele / param->one_core_ele;
  if (total_ele % param->one_core_ele != 0) {
    param->act_core_num = param->act_core_num + 1;
  }
  param->last_core_ele = total_ele - (param->act_core_num - 1) * param->one_core_ele;
}

static void CalTilingMode(MaxPoolV3TilingData* param, const gert::Shape& input_shape,
                          const MaxPoolV3CompileInfo* compile_info, int32_t ksize_h, int32_t ksize_w) {
  int32_t ub_ele = compile_info->ub_ele;
  int32_t core_num = compile_info->core_num;
  int32_t strides_h = compile_info->strides_h;
  int32_t strides_w = compile_info->strides_w;
  OP_TILING_CHECK(strides_h == 0, VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolV3", "strides_h = 0 is unsupported"),
                  return);
  OP_TILING_CHECK(strides_w == 0, VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolV3", "strides_w = 0 is unsupported"),
                  return);

  // calc core_num, core_ele, loop_num and loop_left
  if (ksize_h == param->input_h && ksize_w == param->input_w) {
    param->n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    CalCoreNum(param, param->n_c1, core_num);
    if (ub_ele >= (input_shape.GetDim(INDEX_2) * input_shape.GetDim(INDEX_3) * input_shape.GetDim(INDEX_4))) {
      param->tiling_mode = TILING_MODE_6;
    } else {
      param->h_factor = ub_ele / input_shape.GetDim(INDEX_4);  // acutal is hw_factor
      int32_t input_hw_num = param->input_h * param->input_w;
      param->one_core_loop_num = input_hw_num / param->h_factor;
      // dif from other tiling mode,this is used to tiling hw
      param->one_core_loop_left = input_hw_num % param->h_factor;
      param->last_core_loop_num = param->one_core_loop_num;
      param->last_core_loop_left = param->one_core_loop_left;
      param->tiling_mode = TILING_MODE_7;
    }
    return;
  }
  if ((ksize_h == 1) && (ksize_w == 1) && (strides_h == 1) && (strides_w == 1)) {
    param->tiling_mode = TILING_MODE_0;
    int32_t max_ele = ub_ele / input_shape.GetDim(INDEX_4);
    int32_t total_ele = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1) * input_shape.GetDim(INDEX_2) *
                        input_shape.GetDim(INDEX_3);
    CalCoreNum(param, total_ele, core_num);
    param->one_core_loop_num = param->one_core_ele / max_ele;
    param->one_core_loop_left = param->one_core_ele % max_ele;
    param->last_core_loop_num = param->last_core_ele / max_ele;
    param->last_core_loop_left = param->last_core_ele % max_ele;
  } else {
    int32_t one_sixth_ub_ele = ub_ele / 6;
    param->n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    if (param->pad_h * param->pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param->tiling_mode = TILING_MODE_1;
      CalCoreNum(param, param->n_c1, core_num);
      param->c_factor = one_sixth_ub_ele / (param->pad_h * param->pad_w * input_shape.GetDim(INDEX_4));
      param->one_core_loop_num = param->one_core_ele / param->c_factor;
      param->one_core_loop_left = param->one_core_ele % param->c_factor;
      param->last_core_loop_num = param->last_core_ele / param->c_factor;
      param->last_core_loop_left = param->last_core_ele % param->c_factor;
    } else if (ksize_h * param->pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param->h_factor = (one_sixth_ub_ele / (param->pad_w * input_shape.GetDim(INDEX_4)) - ksize_h) / strides_h + 1;
      int32_t h_loop = param->output_h / param->h_factor;
      if (h_loop <= param->n_c1) {
        param->tiling_mode = TILING_MODE_2;
        CalCoreNum(param, param->n_c1, core_num);
        param->one_core_loop_num = param->output_h / param->h_factor;
        param->one_core_loop_left = param->output_h % param->h_factor;
        param->last_core_loop_num = param->one_core_loop_num;
        param->last_core_loop_left = param->one_core_loop_left;
      } else {
        param->tiling_mode = TILING_MODE_4;
        CalCoreNum(param, param->output_h, core_num);
        param->one_core_loop_num = param->one_core_ele / param->h_factor;
        param->one_core_loop_left = param->one_core_ele % param->h_factor;
        param->last_core_loop_num = param->last_core_ele / param->h_factor;
        param->last_core_loop_left = param->last_core_ele % param->h_factor;
      }
    } else {
      param->w_factor = (one_sixth_ub_ele / input_shape.GetDim(INDEX_4) / ksize_h - ksize_w) / strides_w + 1;
      param->one_core_loop_num = param->output_w / param->w_factor;
      param->one_core_loop_left = param->output_w % param->w_factor;
      param->last_core_loop_num = param->one_core_loop_num;
      param->last_core_loop_left = param->one_core_loop_left;
      if (param->output_h <= param->n_c1) {
        param->tiling_mode = TILING_MODE_3;
        CalCoreNum(param, param->n_c1, core_num);
      } else {
        param->tiling_mode = TILING_MODE_5;
        CalCoreNum(param, param->output_h, core_num);
      }
    }
  }
}

static void CalTilingParam(MaxPoolV3TilingData* param, const gert::Shape& input_shape,
                           const MaxPoolV3CompileInfo* compile_info, int32_t ksize_h, int32_t ksize_w) {
  int32_t strides_h = compile_info->strides_h;
  int32_t strides_w = compile_info->strides_w;
  int32_t padding = compile_info->padding;      // SAME
  int32_t ceil_mode = compile_info->ceil_mode;  // floor
  int32_t pad_top = compile_info->pad_top;
  int32_t pad_bottom = compile_info->pad_bottom;
  int32_t pad_left = compile_info->pad_left;
  int32_t pad_right = compile_info->pad_right;
  OP_TILING_CHECK(strides_h == 0, VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolV3", "strides_h = 0 is unsupported"),
                  return);
  OP_TILING_CHECK(strides_w == 0, VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolV3", "strides_w = 0 is unsupported"),
                  return);
  // calc output height and width, pad infos
  param->c_factor = 1;
  param->h_factor = 1;
  param->w_factor = 1;
  if (padding == 0) {
    param->output_h = (param->input_h + strides_h - 1) / strides_h;
    param->output_w = (param->input_w + strides_w - 1) / strides_w;
    param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
    param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
    param->pad_t = (param->pad_h - param->input_h) / TWICE > 0 ? (param->pad_h - param->input_h) / TWICE : 0;
    param->pad_b = param->pad_h - param->input_h - param->pad_t > 0 ? param->pad_h - param->input_h - param->pad_t : 0;
    param->pad_l = (param->pad_w - param->input_w) / TWICE > 0 ? (param->pad_w - param->input_w) / TWICE : 0;
    param->pad_r = param->pad_w - param->input_w - param->pad_l > 0 ? param->pad_w - param->input_w - param->pad_l : 0;
  } else if (padding == PADDING_VALUE) {
    if (ceil_mode == 1) {
      param->output_h = (param->input_h + pad_top + pad_bottom - ksize_h + strides_h + strides_h - 1) / strides_h;
      param->output_w = (param->input_w + pad_left + pad_right - ksize_w + strides_w + strides_w - 1) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        param->output_h -= ((param->output_h - 1) * strides_h >= param->input_h + pad_top) ? 1 : 0;
        param->output_w -= ((param->output_w - 1) * strides_w >= param->input_w + pad_left) ? 1 : 0;
      }
      param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
      param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
      param->pad_t = pad_top;
      param->pad_b = (param->pad_h - param->input_h - pad_top) > 0 ? (param->pad_h - param->input_h - pad_top) : 0;
      param->pad_l = pad_left;
      param->pad_r = (param->pad_w - param->input_w - pad_left) > 0 ? (param->pad_w - param->input_w - pad_left) : 0;
    } else {
      param->output_h = (param->input_h + pad_top + pad_bottom - ksize_h + strides_h) / strides_h;
      param->output_w = (param->input_w + pad_left + pad_right - ksize_w + strides_w) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        param->output_h -= ((param->output_h - 1) * strides_h >= param->input_h + pad_top) ? 1 : 0;
        param->output_w -= ((param->output_w - 1) * strides_w >= param->input_w + pad_left) ? 1 : 0;
      }
      param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
      param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
      param->pad_t = pad_top;
      param->pad_b = (param->pad_h - param->input_h - pad_top) > 0 ? (param->pad_h - param->input_h - pad_top) : 0;
      param->pad_l = pad_left;
      param->pad_r = (param->pad_w - param->input_w - pad_left) > 0 ? (param->pad_w - param->input_w - pad_left) : 0;
    }
  } else {
    param->output_h = (param->input_h - (ksize_h - 1) + strides_h - 1) / strides_h;
    param->output_w = (param->input_w - (ksize_w - 1) + strides_w - 1) / strides_w;
    param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
    param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
    param->pad_t = 0;
    param->pad_b = 0;
    param->pad_l = 0;
    param->pad_r = 0;
  }

  CalTilingMode(param, input_shape, compile_info, ksize_h, ksize_w);
}

ge::graphStatus TilingForMaxPoolV3(gert::TilingContext* context) {
  auto x_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  auto compile_info = reinterpret_cast<const MaxPoolV3CompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto tilingdata = context->GetTilingData<MaxPoolV3TilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tilingdata);

  // get and check input format and shape
  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  ge::Format input_format = src_td->GetStorageFormat();
  OP_TILING_CHECK(
      input_format != FORMAT_NC1HWC0,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                      "Get input format failed, only support NC1HWC0, but got %d.", input_format),
      return ge::GRAPH_FAILED);

  const auto& input_shape = x_shape->GetStorageShape();
  uint64_t dimnum = input_shape.GetDimNum();
  OP_TILING_CHECK(dimnum != DIMNUM_NC1HWC0,
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      "MaxPoolV3", "Get input shape failed, the length of input shape must be 5, but got %lu.", dimnum),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(input_shape.GetDim(INDEX_4) != DIMSIZE_C0,
                  VECTOR_INNER_ERR_REPORT_TILIING(
                      "MaxPoolV3", "Get input shape failed, dim 5 of input_shape must be 16, but got %lu.",
                      input_shape.GetDim(INDEX_4)),
                  return ge::GRAPH_FAILED);

  // check compile info paramters
  OP_TILING_CHECK((compile_info->ub_ele <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ub_ele must be greater than 0."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK((compile_info->core_num <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "core_num must be greater than 0."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK((compile_info->ksize_h <= 0) || (compile_info->ksize_w <= 0) || (compile_info->strides_h <= 0) ||
                      (compile_info->strides_w <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "ksize and strides must be greater than 0."),
                  return ge::GRAPH_FAILED);

  // check ksize, strides and input shape
  int32_t ksize_h = compile_info->ksize_h;
  int32_t ksize_w = compile_info->ksize_w;
  tilingdata->input_h = input_shape.GetDim(INDEX_2);
  tilingdata->input_w = input_shape.GetDim(INDEX_3);
  if (compile_info->global == 1) {
    ksize_h = tilingdata->input_h;
    ksize_w = tilingdata->input_w;
  }
  OP_TILING_CHECK(
      (compile_info->padding == 1) && ((ksize_h > tilingdata->input_h) || (ksize_w > tilingdata->input_w)),
      VECTOR_INNER_ERR_REPORT_TILIING(
          context->GetNodeName(),
          "Input height or width must greater than or equal to ksize's when padding mode is valid."),
      return ge::GRAPH_FAILED);

  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(tilingdata, input_shape, compile_info, ksize_h, ksize_w);
  if ((compile_info->pad_left > 0) || (compile_info->pad_top > 0)) {
    OP_TILING_CHECK(
        ((tilingdata->output_w - 1) * compile_info->strides_w >= tilingdata->input_w + compile_info->pad_left) ||
            ((tilingdata->output_h - 1) * compile_info->strides_h >= tilingdata->input_h + compile_info->pad_top),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(),
            "Can not ensure that the last pooling starts strictly inside the image even after clip the last."),
        return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMaxPoolV3(gert::TilingParseContext* context) {
  auto compile_info = GetCompileInfoPtr<MaxPoolV3CompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  ReadCompileItem(vars, "ub_ele", compile_info->ub_ele);
  ReadCompileItem(vars, "core_num", compile_info->core_num);
  ReadCompileItem(vars, "ksize_h", compile_info->ksize_h);
  ReadCompileItem(vars, "ksize_w", compile_info->ksize_w);
  ReadCompileItem(vars, "strides_h", compile_info->strides_h);
  ReadCompileItem(vars, "strides_w", compile_info->strides_w);
  ReadCompileItem(vars, "padding", compile_info->padding);
  ReadCompileItem(vars, "ceil_mode", compile_info->ceil_mode);
  ReadCompileItem(vars, "pad_top", compile_info->pad_top);
  ReadCompileItem(vars, "pad_bottom", compile_info->pad_bottom);
  ReadCompileItem(vars, "pad_left", compile_info->pad_left);
  ReadCompileItem(vars, "pad_right", compile_info->pad_right);
  ReadCompileItem(vars, "global", compile_info->global);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolV3).Tiling(TilingForMaxPoolV3).TilingParse<MaxPoolV3CompileInfo>(TilingPrepareForMaxPoolV3);
}  // namespace optiling
