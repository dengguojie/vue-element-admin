/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file conv2d_backprop_input.cpp
 * \brief tiling function of conv2d_backprop_input
 */
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "conv2d_bp_input_cache_tiling.h"
#include "op_tiling.h"
#include "op_log.h"

namespace optiling {
const size_t kConv2dDxInputSizeLimit = 3;
const int64_t kBlockSize = 16;
const int64_t kC0 = 16;
const int64_t kConv2dDimSizeLimit = 4;
// pad
const size_t kConv2dPadUpIdx = 0;
const size_t kConv2dPadDownIdx = 1;
const size_t kConv2dPadLeftIdx = 2;
const size_t kConv2dPadRightIdx = 3;
// NC1HWC0
const size_t kNDimNC1HWC0Idx = 0;
const size_t kC1DimNC1HWC0Idx = 1;
const size_t kHDimNC1HWC0Idx = 2;
const size_t kWDimNC1HWC0Idx = 3;
const size_t kC0DimNC1HWC0Idx = 4;
// NCHW
const size_t kNDimNCHWIdx = 0;
const size_t kCDimNCHWIdx = 1;
const size_t kHDimNCHWIdx = 2;
const size_t kWDimNCHWIdx = 3;
// HWCN
const size_t kNDimHWCNIdx = 3;
const size_t kCDimHWCNIdx = 2;
const size_t kHDimHWCNIdx = 0;
const size_t kWDimHWCNIdx = 1;

const int64_t kDimHWUp = 4096;
const int64_t kDimBatchUp = ((1UL << 31) - 1);
const int64_t kDataSizeMax = ((1UL << 63) - 1);
const int64_t kDimHWLow = 2;
const int64_t kDimLow = 1;
const int64_t kFilterDimHWUp = 255;
const int64_t kStrideDimHWLow = 1;
const int64_t kDilationDimHWUp = 1;
const int64_t kFp16Bytes = 2;
const int64_t kL1size = (1024 * 1024);
const int64_t kStrideHWUp = 63;
const int64_t kConv2dNC1HWC0Size = 5;
const int64_t kConv2dNCHWSize = 4;
const int64_t kInputOutBackpropIndex = 2;

static map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"}, {ge::FORMAT_NHWC, "NHWC"}, {ge::FORMAT_HWCN, "HWCN"}, {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"},
    {ge::FORMAT_NC1HWC0, "NC1HWC0"}};

int64_t Align(const int64_t& param1, const int64_t& param2) {
  if (param2 == 0) {
    return 0;
  }
  return ((param1 + param2 - 1) / param2) * param2;
}

bool CheckRange(const int64_t& value, const int64_t& value_low, const int64_t& value_up) {
  if (value < value_low) {
    return false;
  } else if (value_up != 0 && value > value_up){
    return false;
  }
  return true;
}

bool CheckValue(const int64_t& value, const int64_t& value_temp) {
  if (value != value_temp) {
    return false;
  }
  return true;
}

bool CalculateGroup(const DxParas& dx_paras, map<std::string, int64_t> &group_map) {
  CHECK_OP_FUNC(!CheckValue(dx_paras.co % dx_paras.groups, 0), return false,
                "c dim of dedy must be divisible by groups");
  CHECK_OP_FUNC(!CheckValue(dx_paras.cin % dx_paras.groups, 0), return false,
                "c dim of dedx must be divisible by groups");
  CHECK_OP_FUNC(!CheckValue(dx_paras.cin, dx_paras.kc * dx_paras.groups), return false,
                "c dim of dedx must be equal with filter c multi groups");
  CHECK_OP_FUNC(!CheckValue(dx_paras.co, dx_paras.kn), return false,
                "c dim of dedy must be equal with filter n");
  int64_t dx_c_ori = dx_paras.kc;
  int64_t dy_c_ori = dx_paras.kn / dx_paras.groups;
  int64_t filter_batch_ori = dy_c_ori;
  int64_t filter_c_ori = dx_c_ori;
  int64_t c0 = kC0;
  int64_t dx_c_extend = Lcm(dx_c_ori, c0);
  int64_t dy_c_extend = Lcm(dy_c_ori, c0);
  int64_t multiple_extend = min(Lcm(dx_c_extend, dy_c_extend), dx_paras.groups);
  int64_t dx_c1_extend = multiple_extend * dx_c_ori;
  dx_c1_extend = Lcm(dx_c1_extend, c0);
  int64_t dy_c1_extend = multiple_extend * dy_c_ori;
  dy_c1_extend = Lcm(dy_c1_extend, c0);
  group_map = {{"g_extend", static_cast<int64_t>(
                                ceil(static_cast<double>(dx_paras.groups) / static_cast<double>(multiple_extend)))},
               {"multiple_extend", multiple_extend},
               {"groups", dx_paras.groups},
               {"dx_c1_extend", dx_c1_extend},
               {"dy_c1_extend", dy_c1_extend},
               {"dx_c_ori", dx_c_ori},
               {"dy_c_ori", dy_c_ori},
               {"filter_batch_ori", filter_batch_ori},
               {"filter_c_ori", filter_c_ori}};
  return true;
}

bool CheckShapeRelation(const DxParas& dx_paras) {
  map<std::string, int64_t> group_map;
  CHECK_OP_FUNC(!CalculateGroup(dx_paras, group_map), return false, "Calculate Group invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.batch, dx_paras.batch_o), return false, "dedx batch not eqaul dedy batch");
  CHECK_OP_FUNC(dx_paras.filter_h_dilation > dx_paras.fmap_h_padding, return false,
                "filter_h_dilation or fmap_h_padding invalid");
  CHECK_OP_FUNC(dx_paras.filter_w_dilation > dx_paras.fmap_w_padding, return false,
                "filter_w_dilation or fmap_w_padding invalid");
  return true;
}

bool CheckL1SizeLimit(const DxParas& dx_paras) {
  int64_t w_value = dx_paras.wo * dx_paras.stride_w;
  int64_t h_value_max = 1;
  if (dx_paras.w % kC0 != 0) {
    h_value_max += 1;
  }
  int64_t a_l1_size = h_value_max * w_value * kC0 * kFp16Bytes;
  int64_t b_l1_size = dx_paras.kw * kC0 * dx_paras.kw * kC0 * kFp16Bytes;
  CHECK_OP_FUNC(a_l1_size + b_l1_size > kL1size, return false, "check l1size fail");
  return true;
}

bool CheckParams(const DxParas& dx_paras) {
  CHECK_OP_FUNC(!CheckRange(dx_paras.kh, kDimLow, kFilterDimHWUp), return false, "kh value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.kw, kDimLow, kFilterDimHWUp), return false, "kw value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.batch, kDimLow, kDimBatchUp), return false, "batch value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.co1, kDimLow, 0), return false, "co1 value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.ho, kDimLow, kDimHWUp), return false, "ho value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.wo, kDimLow, kDimHWUp), return false, "wo value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.co, kDimLow, 0), return false, "co value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.c1, kDimLow, 0), return false, "c1 value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.cin, kDimLow, 0), return false, "cin value invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.dilations_n, 1), return false, "dilations_n value invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.dilations_c, 1), return false, "dilations_c value invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.dilations_h, 1), return false, "dilations_h value invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.dilations_w, 1), return false, "dilations_w value invalid");
  CHECK_OP_FUNC(!CheckValue(dx_paras.groups, 1), return false, "groups value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.ho * dx_paras.stride_h, kDimHWLow, kDimHWUp), return false,
                "dedy's H after expands is invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.wo * dx_paras.stride_w, kDimHWLow, kDimHWUp), return false,
                "dedy's W after expands is invalid");
  CHECK_OP_FUNC(!CheckShapeRelation(dx_paras), return false, "check shape relation invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.h, kDimHWLow, kDimHWUp), return false, "h value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.w, kDimHWLow, kDimHWUp), return false, "w value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.stride_h, kDimLow, kStrideHWUp), return false, "stride_h value invalid");
  CHECK_OP_FUNC(!CheckRange(dx_paras.stride_w, kDimLow, kStrideHWUp), return false, "stride_w value invalid");
  CHECK_OP_FUNC((dx_paras.fmap_h_padding - dx_paras.filter_h_dilation) / dx_paras.stride_h + 1 != dx_paras.ho,
                return false, "fmap_h does not match dedy_h");
  CHECK_OP_FUNC((dx_paras.fmap_w_padding - dx_paras.filter_w_dilation) / dx_paras.stride_w + 1 != dx_paras.wo,
                return false, "fmap_w does not match dedy_w");
  CHECK_OP_FUNC(!CheckL1SizeLimit(dx_paras), return false, "this case may excceed L1size");
  int64_t dedy_c_16 = Align(dx_paras.co, kC0);
  CHECK_OP_FUNC(dedy_c_16 == 0, return false, "dedy_c_16 is invalid");
  int64_t dedx_c_16 = Align(dx_paras.cin, kC0);
  CHECK_OP_FUNC(dedx_c_16 == 0, return false, "dedy_c_16 is invalid");
  int64_t filter_c_16 = Align(dx_paras.kc, kC0);
  CHECK_OP_FUNC(filter_c_16 == 0, return false, "dedy_c_16 is invalid");
  int64_t filter_n_16 = Align(dx_paras.kn, kC0);
  CHECK_OP_FUNC(filter_n_16 == 0, return false, "dedy_c_16 is invalid");
  int64_t dedy_size = dx_paras.batch * dedy_c_16 * dx_paras.wo * dx_paras.ho * kFp16Bytes;
  CHECK_OP_FUNC(dedy_size > kDataSizeMax, return false, "dedy size large than int64");
  int64_t dedx_size = dx_paras.batch * dedx_c_16 * dx_paras.w * dx_paras.h * kFp16Bytes;
  CHECK_OP_FUNC(dedx_size > kDataSizeMax, return false, "dedx size large than int64");
  int64_t filter_size = filter_n_16 * filter_c_16 * dx_paras.kw * dx_paras.kh * kFp16Bytes;
  CHECK_OP_FUNC(filter_size > kDataSizeMax, return false, "filter size large than int64");
  return true;
}

bool GetAttrFromOp(const ge::OpDescPtr& op_desc, DxParas& dx_paras) {
  std::vector<int64_t> strides_list;
  std::vector<int64_t> pads_list;
  std::vector<int64_t> dilations_list;
  std::string data_format;
  CHECK_OP_FUNC(!ge::AttrUtils::GetListInt(op_desc, "strides", strides_list), return false, "get strides failed");
  CHECK_OP_FUNC(!ge::AttrUtils::GetInt(op_desc, "groups", dx_paras.groups), return false, "get groups failed");
  CHECK_SIZE(strides_list.size() != kConv2dDimSizeLimit, return false, "strides is invalid");
  CHECK_OP_FUNC(!ge::AttrUtils::GetListInt(op_desc, "pads", pads_list), return false, "get pads failed");
  CHECK_SIZE(pads_list.size() != kConv2dDimSizeLimit, return false, "pads is invalid");
  CHECK_OP_FUNC(!ge::AttrUtils::GetListInt(op_desc, "dilations", dilations_list), return false,
                "get dilations failed");
  CHECK_SIZE(dilations_list.size() != kConv2dDimSizeLimit, return false, "dilations is invalid");
  CHECK_OP_FUNC(!ge::AttrUtils::GetStr(op_desc, "data_format", data_format), return false, "get data_format failed");
  CHECK_SIZE(data_format.length() != kConv2dDimSizeLimit, return false, "the format is not 4D");
  CHECK_OP_FUNC(data_format != "NCHW", return false, "data_format is not NCHW");
  dx_paras.padu = pads_list[kConv2dPadUpIdx];
  dx_paras.padd = pads_list[kConv2dPadDownIdx];
  dx_paras.padl = pads_list[kConv2dPadLeftIdx];
  dx_paras.padr = pads_list[kConv2dPadRightIdx];
  dx_paras.stride_h = strides_list[kHDimNCHWIdx];
  dx_paras.stride_w = strides_list[kWDimNCHWIdx];
  dx_paras.dilations_n = dilations_list[kNDimNCHWIdx];
  dx_paras.dilations_c = dilations_list[kCDimNCHWIdx];
  dx_paras.dilations_h = dilations_list[kHDimNCHWIdx];
  dx_paras.dilations_w = dilations_list[kWDimNCHWIdx];
  return true;
}

bool GetAttrFromCompileInfo(const nlohmann::json& compile_info, DxParas& dx_paras) {
  bool compile_info_invalid = !compile_info.contains("attrs") || !compile_info["attrs"].contains("strides") ||
                              !compile_info["attrs"].contains("pads") || !compile_info["attrs"].contains("groups") ||
                              !compile_info["attrs"].contains("dilations") ||
                              !compile_info["attrs"].contains("data_format") ||
                              compile_info["attrs"]["strides"].size() != kConv2dNCHWSize ||
                              compile_info["attrs"]["pads"].size() != kConv2dNCHWSize ||
                              compile_info["attrs"]["dilations"].size() != kConv2dNCHWSize ||
                              compile_info["attrs"]["groups"].size() != 1;
  CHECK_OP_FUNC(compile_info_invalid, return false, "get attr failed from compile_info");
  CHECK_OP_FUNC(compile_info["attrs"]["data_format"] != "NCHW", return false, "data_format is not NCHW");
  dx_paras.groups = compile_info["attrs"]["groups"];
  std::vector<int64_t> strides_list = compile_info["attrs"]["strides"];
  std::vector<int64_t> pads_list = compile_info["attrs"]["pads"];
  std::vector<int64_t> dilations_list = compile_info["attrs"]["dilations"];
  dx_paras.padu = pads_list[kConv2dPadUpIdx];
  dx_paras.padd = pads_list[kConv2dPadDownIdx];
  dx_paras.padl = pads_list[kConv2dPadLeftIdx];
  dx_paras.padr = pads_list[kConv2dPadRightIdx];
  dx_paras.stride_h = strides_list[kHDimNCHWIdx];
  dx_paras.stride_w = strides_list[kWDimNCHWIdx];
  dx_paras.dilations_n = dilations_list[kNDimNCHWIdx];
  dx_paras.dilations_c = dilations_list[kCDimNCHWIdx];
  dx_paras.dilations_h = dilations_list[kHDimNCHWIdx];
  dx_paras.dilations_w = dilations_list[kWDimNCHWIdx];
  return true;
}

bool Conv2DBackpropInputParseFunc(const ge::Operator& op_paras, const nlohmann::json& compile_info,
                                  DxParas& dx_paras) {
  if (compile_info.contains("tiling_type") && compile_info["tiling_type"] == "binary") {
    dx_paras.repo_binary_flag = true;
    ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    ge::AscendString op_type;
    CHECK_OP_FUNC(op_paras.GetOpType(op_type) != ge::GRAPH_SUCCESS, return false, "failed to get op_type");
    dx_paras.op_type = string(op_type.GetString());
    if (!GetAttrFromOp(op_desc, dx_paras)) {
      CHECK_OP_FUNC(!GetAttrFromCompileInfo(compile_info, dx_paras), return false, "get attr failed");
    }
    CHECK_OP_FUNC(!compile_info.contains("block_dim") || !compile_info["block_dim"].contains("CORE_NUM"),
                  return false, "get core_num failed");
    dx_paras.core_num = compile_info["block_dim"]["CORE_NUM"];
    ge::GeTensorDescPtr filter_desc = op_desc->MutableInputDesc(1);
    CHECK_OP_FUNC(filter_desc == nullptr, return false, "tensor filter desc failed");
    ge::GeTensorDescPtr out_backprop_desc = op_desc->MutableInputDesc(kInputOutBackpropIndex);
    CHECK_OP_FUNC(out_backprop_desc == nullptr, return false, "tensor out_backprop desc failed");
    ge::GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
    CHECK_OP_FUNC(y_desc == nullptr, return false, "tensor y desc failed");
    dx_paras.batch = out_backprop_desc->GetShape().GetDim(kNDimNC1HWC0Idx);
    dx_paras.batch_o = y_desc->GetShape().GetDim(kNDimNC1HWC0Idx);
    CHECK_OP_FUNC(y_desc->GetOriginFormat() != ge::FORMAT_NCHW, return false, "y ori_format invalid");
    CHECK_OP_FUNC(out_backprop_desc->GetOriginFormat() != ge::FORMAT_NCHW, return false,
                  "out_backprop ori_format failed");
    CHECK_OP_FUNC(filter_desc->GetFormat() != ge::FORMAT_FRACTAL_Z, return false, "filter format failed");
    CHECK_OP_FUNC(
        filter_desc->GetOriginFormat() != ge::FORMAT_NCHW && filter_desc->GetOriginFormat() != ge::FORMAT_HWCN,
        return false, "filter ori_format failed");
    CHECK_OP_FUNC(y_desc->GetFormat() != ge::FORMAT_NCHW, return false, "y format invalid");

    if (dx_paras.stride_h == 1 && dx_paras.stride_w == 1) {
      dx_paras.stride_expand_flag = 0;
      CHECK_SIZE(out_backprop_desc->GetShape().GetDimNum() != kConv2dNCHWSize,
          return false, "out_backprop shape len is invalid");
      CHECK_OP_FUNC(out_backprop_desc->GetFormat() != ge::FORMAT_NCHW, return false, "out_backprop format failed");
    } else {
      dx_paras.stride_expand_flag = 1;
      CHECK_SIZE(out_backprop_desc->GetShape().GetDimNum() != kConv2dNC1HWC0Size,
          return false, "out_backprop shape len is invalid");
      CHECK_OP_FUNC(out_backprop_desc->GetFormat() != ge::FORMAT_NC1HWC0, return false, "out_backprop format failed");
    }

    std::string y_ori_format = format2str[y_desc->GetOriginFormat()];
    CHECK_SIZE(out_backprop_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize,
               return false, "out_backprop ori_shape len is invalid");
    dx_paras.co = out_backprop_desc->GetOriginShape().GetDim(y_ori_format.find("C"));
    dx_paras.co1 = (dx_paras.co + kBlockSize - 1) / kBlockSize;
    dx_paras.ho = out_backprop_desc->GetShape().GetDim(kHDimNC1HWC0Idx);
    dx_paras.wo = out_backprop_desc->GetShape().GetDim(kWDimNC1HWC0Idx);
    CHECK_SIZE(filter_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize,
               return false, "filter ori_shape len is invalid");
    CHECK_SIZE(filter_desc->GetShape().GetDimNum() != kConv2dNCHWSize,  return false, "filter shape len is invalid");
    dx_paras.filter_cin1hw = filter_desc->GetShape().GetDim(0);
    dx_paras.filter_cout1 = filter_desc->GetShape().GetDim(1);
    if (filter_desc->GetOriginFormat() == ge::FORMAT_NCHW) {
      dx_paras.kn = filter_desc->GetOriginShape().GetDim(kNDimNCHWIdx);
      dx_paras.kc = filter_desc->GetOriginShape().GetDim(kCDimNCHWIdx);
      dx_paras.kh = filter_desc->GetOriginShape().GetDim(kHDimNCHWIdx);
      dx_paras.kw = filter_desc->GetOriginShape().GetDim(kWDimNCHWIdx);
    } else {
      dx_paras.kn = filter_desc->GetOriginShape().GetDim(kNDimHWCNIdx);
      dx_paras.kc = filter_desc->GetOriginShape().GetDim(kCDimHWCNIdx);
      dx_paras.kh = filter_desc->GetOriginShape().GetDim(kHDimHWCNIdx);
      dx_paras.kw = filter_desc->GetOriginShape().GetDim(kWDimHWCNIdx);
    }

    CHECK_SIZE(y_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize, return false, "y ori_shape len is invalid");
    CHECK_SIZE(y_desc->GetShape().GetDimNum() != kConv2dNCHWSize, return false, "y shape len is invalid");
    dx_paras.cin = y_desc->GetOriginShape().GetDim(y_ori_format.find("C"));
    dx_paras.c1 = (dx_paras.cin + kBlockSize - 1) / kBlockSize;
    dx_paras.h = y_desc->GetShape().GetDim(kHDimNC1HWC0Idx);
    dx_paras.w = y_desc->GetShape().GetDim(kWDimNC1HWC0Idx);
    dx_paras.fmap_h_padding = dx_paras.h + dx_paras.padu + dx_paras.padd;
    dx_paras.fmap_w_padding = dx_paras.w + dx_paras.padl + dx_paras.padr;
    dx_paras.filter_h_dilation = (dx_paras.kh - 1) * dx_paras.dilations_h + 1;
    dx_paras.filter_w_dilation = (dx_paras.kw - 1) * dx_paras.dilations_w + 1;
  }
  return true;
}

/*
 * @brief: tiling function of conv2d_backprop_input
 * @param [in] op_type: op_type of conv2d_backprop_input
 * @param [in] op_paras: inputs/outputs/atts of conv2d_backprop_input
 * @param [in] op_compile_info: compile time generated info of conv2d_backprop_input
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DBpInputTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                         utils::OpRunInfo& runInfo) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  CHECK_OP_FUNC(op_desc == nullptr, return false, "the op_desc is nullptr.");
  // the input tensor's index is 2
  ge::ConstGeTensorDescPtr tensor_in_desc = op_desc->GetInputDescPtr(2);
  ge::ConstGeTensorDescPtr tensor_out_desc = op_desc->GetOutputDescPtr(0);
  const ge::GeShape &tensor_in_shape = tensor_in_desc->GetShape();
  const ge::GeShape &tensor_out_shape = tensor_out_desc->GetShape();
  size_t output_dimnum = tensor_out_shape.GetDimNum();
  bool unvalid_size = opParas.GetInputsSize() < kConv2dDxInputSizeLimit || opParas.GetOutputsSize() == 0 ||
                      tensor_in_shape.GetDimNum() < kConv2dDimNumLimit || output_dimnum < kConv2dDimNumLimit;
  CHECK_OP_FUNC(unvalid_size, return false, "the size is unvalid.");
  GELOGD("Current format is %s, Ori format is %s",
         ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetFormat()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetOriginFormat()).c_str());

  try {
    CHECK_OP_FUNC(opCompileInfo.empty(), return false, "op compile info is empty");
    // accurate build has only one item
    // fuzzy build has multiple items
    std::vector<std::string> varMap;
    nlohmann::json opInfo;
    GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
    DxParas dx_paras;
    if (opCompileInfo.is_array()) {
      // >>> start: splice compile info
      opInfo = opCompileInfo[0];
      varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();
      nlohmann::json item;
      for (size_t i = 1; i < opCompileInfo.size(); ++i) {
        item = opCompileInfo[i];
        std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
        for (auto &key : key_list) {
          auto &item_key = item[key];
          if (item_key.is_object() && !item_key.empty()) {
            std::vector<int32_t> list_value = item_key.begin().value().get<std::vector<int32_t>>();
            opInfo[key][item_key.begin().key()] = list_value;
          }
        }
        std::string key_int = "block_dim";
        auto &item_key_int = item[key_int];
        if (item_key_int.is_object() && !item_key_int.empty()) {
          int32_t int_value = item_key_int.begin().value().get<int32_t>();
          opInfo[key_int][item_key_int.begin().key()] = int_value;
        }
      }
      // <<< end: put together compile info
      GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
    } else if (opCompileInfo.is_object()) {
      CHECK_OP_FUNC(!Conv2DBackpropInputParseFunc(opParas, opCompileInfo, dx_paras), return false, "ParseFunc failed!");
      if (dx_paras.repo_binary_flag) {
        string tiling_id;
        Tiling tiling;
        CHECK_OP_FUNC(!CheckParams(dx_paras), return false, "Check failed!");
        CHECK_OP_FUNC(!GenTiling(dx_paras, tiling, tiling_id), return false, "GenTiling failed!");
        CHECK_OP_FUNC(!UpdateRunInfoBinary(dx_paras, tiling, tiling_id, runInfo), return false,
                      "UpdateRunInfo failed!");
        return true;
      }
      varMap = opCompileInfo.at("_vars")["10000"].get<std::vector<std::string>>();
      opInfo = opCompileInfo;
    }

    std::vector<int64_t> var_value;
    if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dNDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_in_shape.GetDim(kConv2dHDim));
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dHDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
      var_value.insert(var_value.end(), tensor_in_shape.GetDim(kConv2dWDim));
      var_value.insert(var_value.end(), tensor_out_shape.GetDim(kConv2dWDim));
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(output_dimnum);
    for (size_t i = 0; i < output_dimnum; i++) {
      output_shape.emplace_back(tensor_out_shape.GetDim(i));
    }
    return cube_tiling(opType, output_shape, var_value, opInfo, runInfo);
  } catch (...) {
    GELOGD("get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the conv2d_backprop_input
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2DBackpropInput, Conv2DBpInputTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DepthwiseConv2DBackpropInput, Conv2DBpInputTiling);
}  // namespace optiling
