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
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "cube_tiling_runtime.h"
#include "graph/debug/ge_log.h"
#include "graph/ge_tensor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "conv2d_bp_input_cache_tiling.h"
#include "op_tiling.h"
#include "op_log.h"
#include "error_log.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"

using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_HWCN;
using ge::FORMAT_NC1HWC0;
using ge::FORMAT_FRACTAL_Z;

namespace optiling {
const size_t kConv2dDxInputSizeLimit = 3;
const int32_t kBlockSize = 16;
const int32_t kC0 = 16;
const int32_t kConv2dDimSizeLimit = 4;
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
// NHWC
const size_t kNDimNHWCIdx = 0;
const size_t kCDimNHWCIdx = 3;
const size_t kHDimNHWCIdx = 1;
const size_t kWDimNHWCIdx = 2;

const int32_t kDimHWUp = 4096;
const int32_t kDimBatchUp = ((1UL << 31) - 1);
const int64_t kDataSizeMax = ((1UL << 63) - 1);
const int32_t kDimHWLow = 2;
const int32_t kDimLow = 1;
const int32_t kFilterDimHWUp = 255;
const int32_t kStrideDimHWLow = 1;
const int32_t kDilationDimHWUp = 1;
const int32_t kFp16Bytes = 2;
const int32_t kUbSize = 262000;
const int32_t kL1size = (1024 * 1024);
const int32_t kStrideHWUp = 63;
const int32_t kConv2dNC1HWC0Size = 5;
const int32_t kConv2dNCHWSize = 4;
const int32_t kInputIndexTwo = 2;
const int32_t kDefaultDilations = 1;
const int32_t kNumTwo = 2;
const int32_t KBinaryModeNC1HWC0 = 1;
const int32_t KBinaryModeNCHW = 2;

inline int32_t Align(const int32_t& param1, const int32_t& param2) {
  if (param2 == 0) {
    return 0;
  }
  return ((param1 + param2 - 1) / param2) * param2;
}

inline bool CheckRange(const int32_t& value, const int32_t& value_low, const int32_t& value_up) {
  if (value < value_low || value > value_up) {
    return false;
  }
  return true;
}

inline bool CheckLowerBound(const int32_t& value, const int32_t& value_low) {
  return value >= value_low;
}

inline bool CheckValue(const int32_t& value, const int32_t& value_temp) {
  return value == value_temp;
}

bool CheckL1SizeLimit(const DxParas& dx_paras) {
  int32_t w_value = dx_paras.wo * dx_paras.stride_w;
  int32_t h_value_max = 1;
  if (dx_paras.w % kC0 != 0) {
    h_value_max += 1;
  }
  int32_t a_l1_size = h_value_max * w_value * kC0 * kFp16Bytes;
  int32_t b_l1_size = dx_paras.kh * kC0 * dx_paras.kw * kC0 * kFp16Bytes;
  CHECK_OP_FUNC(a_l1_size + b_l1_size > kL1size, return false,
                "check l1size failed, a_l1_size is %d, b_l1_size is %d, kL1size is %d", a_l1_size, b_l1_size, kL1size);
  return true;
}

bool CheckUBSizeLimit(const DxParas& dx_paras) {
  int32_t m_aub = 1;
  int32_t k_aub = 1;
  int32_t n_cub = 1;
  int32_t m_l0 = 1;
  int32_t loadin_size = dx_paras.aub_num * k_aub * m_aub * dx_paras.wo * kC0 * dx_paras.stride_w;
  int32_t copyout_size = dx_paras.cub_num * n_cub * m_l0 * kC0 * kC0;
  if (dx_paras.stride_expand_flag == 0) {
    int32_t wo_aub = 1;
    loadin_size = dx_paras.aub_num * k_aub * kBlockSize *
                  ((m_aub * wo_aub + dx_paras.kw - 1 + kBlockSize - 1) / kBlockSize) * kBlockSize;
  }
  int32_t ub_fp16_size = kUbSize / kFp16Bytes;
  CHECK_OP_FUNC(loadin_size + copyout_size > ub_fp16_size, return false,
                "check ubsize fail, loadin_size is %d, copyout_size is %d, ub_fp16_size is %d", loadin_size,
                copyout_size, ub_fp16_size);
  return true;
}

inline void UpdateOpDescAttr(const ge::OpDescPtr& op_desc, ge::OpDescPtr& op_desc_attr) {
  ge::ComputeGraphPtr ori_graph = nullptr;
  if (ge::AttrUtils::GetGraph(op_desc, "_original_fusion_graph", ori_graph)) {
    for (auto &node : ori_graph->GetAllNodes()) {
      if (node->GetType() == "Conv2DBackpropInput") {
        op_desc_attr = node->GetOpDesc();
        break;
      }
    }
  } else {
    GELOGD("this is not fusion node, only conv2d_backprop_input single node");
  }
}

inline string IntToBinary(uint64_t& n) {
  string ans = "";
  do {
    uint64_t t = n % 2UL;
    ans += (t + '0');
    n /= 2UL;
  } while (n);
  return ans;
}

inline void OutputErrorMsg(const string error_info[], string& error_flag) {
  string msg;
  for (size_t i = 0; i < error_flag.length(); i++) {
    if (error_flag[i] == '1') {
      msg = error_info[i];
      OP_LOGE("Conv2DBackpropInput", "Error msg is: %s", msg.c_str());
      break;
    }
  }
}

inline int32_t Lcm(const int32_t &param1, const int32_t &param2) {
  int32_t pram1_lcm = param1;
  int32_t pram2_lcm = param2;
  int32_t temp = pram1_lcm * pram2_lcm;
  int32_t param1_temp = pram1_lcm;
  while (pram1_lcm % pram2_lcm != 0) {
    param1_temp = pram1_lcm;
    pram1_lcm = pram2_lcm;
    pram2_lcm = param1_temp % pram2_lcm;
  }
  return temp / pram2_lcm;
}

bool CheckParams(const DxParas& dx_paras) {
  int32_t dedy_c_16 = Align(dx_paras.co, kC0);
  int32_t dedx_c_16 = Align(dx_paras.cin, kC0);
  int32_t filter_c_16 = Align(dx_paras.kc, kC0);
  int32_t filter_n_16 = Align(dx_paras.kn, kC0);
  int64_t dedy_size = dx_paras.batch * dedy_c_16 * dx_paras.wo * dx_paras.ho * kFp16Bytes;
  int64_t dedx_size = dx_paras.batch * dedx_c_16 * dx_paras.w * dx_paras.h * kFp16Bytes;
  int64_t filter_size = filter_n_16 * filter_c_16 * dx_paras.kw * dx_paras.kh * kFp16Bytes;
  uint32_t shift = 0;
  uint64_t invalid = (!CheckRange(dx_paras.kh, kDimLow, kFilterDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.kw, kDimLow, kFilterDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.batch, kDimLow, kDimBatchUp) << shift++);
  invalid = invalid + (!CheckLowerBound(dx_paras.co1, kDimLow) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.ho, kDimLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.wo, kDimLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckLowerBound(dx_paras.co, kDimLow) << shift++);
  invalid = invalid + (!CheckLowerBound(dx_paras.c1, kDimLow) << shift++);
  invalid = invalid + (!CheckLowerBound(dx_paras.cin, kDimLow) << shift++);
  invalid =
      invalid + ((dx_paras.dilations_h != kDefaultDilations || dx_paras.dilations_w != kDefaultDilations) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.groups, 1) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.ho * dx_paras.stride_h, kDimHWLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.wo * dx_paras.stride_w, kDimHWLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.co % dx_paras.groups, 0) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.cin % dx_paras.groups, 0) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.cin, dx_paras.kc * dx_paras.groups) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.co, dx_paras.kn) << shift++);
  invalid = invalid + (!CheckValue(dx_paras.batch, dx_paras.batch_o) << shift++);
  invalid = invalid + ((dx_paras.filter_h_dilation > dx_paras.fmap_h_padding) << shift++);
  invalid = invalid + ((dx_paras.filter_w_dilation > dx_paras.fmap_w_padding) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.h, kDimHWLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.w, kDimHWLow, kDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.stride_h, kDimLow, kStrideHWUp) << shift++);
  invalid = invalid + (!CheckRange(dx_paras.stride_w, kDimLow, kStrideHWUp) << shift++);
  int32_t ho_temp = (dx_paras.fmap_h_padding - dx_paras.filter_h_dilation) / dx_paras.stride_h + 1;
  int32_t wo_temp = (dx_paras.fmap_w_padding - dx_paras.filter_w_dilation) / dx_paras.stride_w + 1;
  invalid = invalid + ((ho_temp != dx_paras.ho) << shift++);
  invalid = invalid + ((wo_temp != dx_paras.wo) << shift++);
  invalid = invalid + ((dedy_c_16 == 0) << shift++);
  invalid = invalid + ((dedx_c_16 == 0) << shift++);
  invalid = invalid + ((filter_c_16 == 0) << shift++);
  invalid = invalid + ((filter_n_16 == 0) << shift++);
  invalid = invalid + ((dedy_size > kDataSizeMax) << shift++);
  invalid = invalid + ((dedx_size > kDataSizeMax) << shift++);
  invalid = invalid + (static_cast<int64_t>((filter_size > kDataSizeMax)) << shift++);
  invalid = invalid + (static_cast<int64_t>(!CheckL1SizeLimit(dx_paras)) << shift++);
  invalid = invalid + (static_cast<int64_t>(!CheckUBSizeLimit(dx_paras)) << shift++);
  if (invalid != 0) {
    string error_info[shift++] = {"kh value invalid", "kw value invalid", "batch value invalid", "co1 value invalid",
                                  "ho value invalid", "wo value invalid", "co value invalid", "c1 value invalid",
                                  "cin value invalid", "dilations value invalid", "groups value invalid",
                                  "dedy's H after expands is invalid", "dedy's W after expands is invalid",
                                  "c dim of dedy must be div by groups", "c dim of dedx must be div by groups",
                                  "c dim of dedx must be equal with filter c multi groups",
                                  "c dim of dedy must be equal with filter n", "dedx batch not equal with dedy batch",
                                  "filter_h_dilation or fmap_h_padding invalid",
                                  "filter_w_dilation or fmap_w_padding invalid", "h value invalid", "w value invalid",
                                  "stride_h invalid", "stride_w invalid", "fmap_h does not match dedy_h",
                                  "fmap_w does not match dedy_w", "dedy_c_16 is invalid", "dedx_c_16 is invalid",
                                  "filter_c_16 is invalid", "filter_n_16 is invalid", "dedy size large than int64",
                                  "dedx size large than int64", "filter size large than int64",
                                  "this case may exceed L1size", "this case may exceed UBsize"};
    string error_flag = IntToBinary(invalid);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }
  return true;
}

bool GetAttrFromOp(const ge::OpDescPtr& op_desc, DxParas& dx_paras) {
  std::vector<int32_t> strides_list;
  std::vector<int32_t> pads_list;
  std::vector<int32_t> dilations_list;
  uint32_t shift = 0;
  uint32_t get_attr_success = (!ge::AttrUtils::GetListInt(op_desc, "strides", strides_list) << shift++);
  get_attr_success = get_attr_success + ((strides_list.size() != kConv2dDimSizeLimit) << shift++);
  get_attr_success = get_attr_success + (!ge::AttrUtils::GetInt(op_desc, "groups", dx_paras.groups) << shift++);
  get_attr_success = get_attr_success + (!ge::AttrUtils::GetListInt(op_desc, "pads", pads_list) << shift++);
  get_attr_success = get_attr_success + ((pads_list.size() != kConv2dDimSizeLimit) << shift++);
  get_attr_success = get_attr_success + (!ge::AttrUtils::GetListInt(op_desc, "dilations", dilations_list) << shift++);
  get_attr_success = get_attr_success + ((dilations_list.size() != kConv2dDimSizeLimit) << shift++);
  if (get_attr_success != 0) {
    string error_info[shift++] = {"get strides failed", "strides is invalid", "get groups failed", "get pads failed",
                                  "pads is invalid", "get dilations failed", "dilations is invalid"};
    uint64_t error_num = static_cast<uint64_t>(get_attr_success);
    string error_flag = IntToBinary(error_num);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }
  ge::ConstGeTensorDescPtr y_desc = op_desc->GetOutputDescPtr(0);
  auto y_ori_format = y_desc->GetOriginFormat();
  if (y_ori_format == ge::FORMAT_NCHW) {
    dx_paras.stride_h = strides_list[kHDimNCHWIdx];
    dx_paras.stride_w = strides_list[kWDimNCHWIdx];
    dx_paras.dilations_n = dilations_list[kNDimNCHWIdx];
    dx_paras.dilations_c = dilations_list[kCDimNCHWIdx];
    dx_paras.dilations_h = dilations_list[kHDimNCHWIdx];
    dx_paras.dilations_w = dilations_list[kWDimNCHWIdx];
  } else {
    dx_paras.stride_h = strides_list[kHDimNHWCIdx];
    dx_paras.stride_w = strides_list[kWDimNHWCIdx];
    dx_paras.dilations_n = dilations_list[kNDimNHWCIdx];
    dx_paras.dilations_c = dilations_list[kCDimNHWCIdx];
    dx_paras.dilations_h = dilations_list[kHDimNHWCIdx];
    dx_paras.dilations_w = dilations_list[kWDimNHWCIdx];
  }
  dx_paras.padu = pads_list[kConv2dPadUpIdx];
  dx_paras.padd = pads_list[kConv2dPadDownIdx];
  dx_paras.padl = pads_list[kConv2dPadLeftIdx];
  dx_paras.padr = pads_list[kConv2dPadRightIdx];

  return true;
}

void CalPads(DxParas& dx_paras) {
  // when padding is SAME, pads is [-1, -1, -1, -1]
  // when padding is VALID, pads is [0, 0, 0, 0]
  if (dx_paras.padu == -1) {
    int32_t pad_h = max(Align(dx_paras.h, dx_paras.stride_h) - dx_paras.stride_h + dx_paras.kh - dx_paras.h, 0);
    int32_t pad_up = (pad_h >> 1L);
    int32_t pad_down = pad_h - pad_up;
    int32_t pad_w = max(Align(dx_paras.w, dx_paras.stride_w) - dx_paras.stride_w + dx_paras.kw - dx_paras.w, 0);
    int32_t pad_left = (pad_w >> 1L);
    int32_t pad_right = pad_w - pad_left;
    dx_paras.padu = pad_up;
    dx_paras.padd = pad_down;
    dx_paras.padl = pad_left;
    dx_paras.padr = pad_right;
  }
}

void CalShapeInfo(DxParas& dx_paras) {
  dx_paras.fmap_h_padding = dx_paras.h + dx_paras.padu + dx_paras.padd;
  dx_paras.fmap_w_padding = dx_paras.w + dx_paras.padl + dx_paras.padr;
  dx_paras.filter_h_dilation = (dx_paras.kh - 1) * dx_paras.dilations_h + 1;
  dx_paras.filter_w_dilation = (dx_paras.kw - 1) * dx_paras.dilations_w + 1;
  dx_paras.dy_c_ori = ((dx_paras.co + dx_paras.groups - 1) / dx_paras.groups) * dx_paras.groups;
  int32_t block_size = kBlockSize;
  int32_t dx_c_extend = Lcm(dx_paras.cin, block_size) / dx_paras.cin;
  int32_t dy_c_extend = Lcm(dx_paras.dy_c_ori, block_size) / dx_paras.dy_c_ori;
  int32_t c_lcm = Lcm(dx_c_extend, dy_c_extend);
  dx_paras.multiple_extend = min(c_lcm, dx_paras.groups);
  dx_paras.g_extend = (dx_paras.groups + dx_paras.multiple_extend - 1) / dx_paras.multiple_extend;
  dx_paras.dx_c1_extend = (dx_paras.multiple_extend * dx_paras.cin + kBlockSize - 1) / kBlockSize;
  dx_paras.pad_up_before = (dx_paras.kh - 1) * dx_paras.dilations_h - dx_paras.padu;
  dx_paras.pad_left_before = (dx_paras.kw - 1) * dx_paras.dilations_w - dx_paras.padl;
  dx_paras.pad_down_after =
      dx_paras.h - dx_paras.ho * dx_paras.stride_h - dx_paras.pad_up_before + (dx_paras.kh - 1) * dx_paras.dilations_h;
  dx_paras.pad_right_after =
      dx_paras.w - dx_paras.wo * dx_paras.stride_w -
      dx_paras.pad_left_before + (dx_paras.kw - 1) * dx_paras.dilations_w;
  dx_paras.shape_up_modify = (dx_paras.pad_up_before - abs(dx_paras.pad_up_before)) / kNumTwo;
  dx_paras.shape_left_modify =
      (dx_paras.pad_left_before - abs(dx_paras.pad_left_before)) / kNumTwo;
  dx_paras.shape_down_modify =
      (dx_paras.pad_down_after - abs(dx_paras.pad_down_after)) / kNumTwo;
  dx_paras.shape_right_modify =
      (dx_paras.pad_right_after - abs(dx_paras.pad_right_after)) / kNumTwo;
  dx_paras.pad_up_before = (dx_paras.pad_up_before + abs(dx_paras.pad_up_before)) / kNumTwo;
  dx_paras.pad_left_before =
      (dx_paras.pad_left_before + abs(dx_paras.pad_left_before)) / kNumTwo;
  dx_paras.pad_down_after = (dx_paras.pad_down_after + abs(dx_paras.pad_down_after)) / kNumTwo;
  dx_paras.pad_right_after =
      (dx_paras.pad_right_after + abs(dx_paras.pad_right_after)) / kNumTwo;
}

void GetInfo(DxParas &dx_paras, const nlohmann::json &compile_info) {
  if (compile_info.contains("binary_mode")) {
    dx_paras.binary_mode = compile_info["binary_mode"];
  }
  if (compile_info.contains("aub_num")) {
    dx_paras.aub_num = compile_info["aub_num"];
  }
  if (compile_info.contains("cub_num")) {
    dx_paras.cub_num = compile_info["cub_num"];
  }
  if (compile_info.contains("ub_size")) {
    dx_paras.ub_size = compile_info["ub_size"];
  }
}

void CalShapeInfoFromDesc(DxParas &dx_paras, const ge::ConstGeTensorDescPtr &filter_desc,
                          const ge::ConstGeTensorDescPtr &out_backprop_desc,
                          const ge::ConstGeTensorDescPtr &y_desc) {
  dx_paras.batch = out_backprop_desc->GetShape().GetDim(kNDimNC1HWC0Idx);
  dx_paras.batch_o = y_desc->GetShape().GetDim(kNDimNC1HWC0Idx);
  dx_paras.ho = out_backprop_desc->GetShape().GetDim(kHDimNC1HWC0Idx);
  dx_paras.wo = out_backprop_desc->GetShape().GetDim(kWDimNC1HWC0Idx);
  dx_paras.filter_cin1hw = filter_desc->GetShape().GetDim(0);
  dx_paras.filter_cout1 = filter_desc->GetShape().GetDim(1);
  auto filter_ori_format = filter_desc->GetOriginFormat();
  if (filter_ori_format == ge::FORMAT_NCHW) {
    dx_paras.kn = filter_desc->GetOriginShape().GetDim(kNDimNCHWIdx);
    dx_paras.kc = filter_desc->GetOriginShape().GetDim(kCDimNCHWIdx);
    dx_paras.kh = filter_desc->GetOriginShape().GetDim(kHDimNCHWIdx);
    dx_paras.kw = filter_desc->GetOriginShape().GetDim(kWDimNCHWIdx);
  } else if (filter_ori_format == ge::FORMAT_HWCN) {
    dx_paras.kn = filter_desc->GetOriginShape().GetDim(kNDimHWCNIdx);
    dx_paras.kc = filter_desc->GetOriginShape().GetDim(kCDimHWCNIdx);
    dx_paras.kh = filter_desc->GetOriginShape().GetDim(kHDimHWCNIdx);
    dx_paras.kw = filter_desc->GetOriginShape().GetDim(kWDimHWCNIdx);
  } else {
    dx_paras.kn = filter_desc->GetOriginShape().GetDim(kNDimNHWCIdx);
    dx_paras.kc = filter_desc->GetOriginShape().GetDim(kCDimNHWCIdx);
    dx_paras.kh = filter_desc->GetOriginShape().GetDim(kHDimNHWCIdx);
    dx_paras.kw = filter_desc->GetOriginShape().GetDim(kWDimNHWCIdx);
  }
  if (y_desc->GetOriginFormat() == ge::FORMAT_NCHW) {
    dx_paras.cin = y_desc->GetOriginShape().GetDim(kCDimNCHWIdx);
  } else {
    dx_paras.cin = y_desc->GetOriginShape().GetDim(kCDimNHWCIdx);
  }
  if (out_backprop_desc->GetOriginFormat() == ge::FORMAT_NCHW) {
    dx_paras.co = out_backprop_desc->GetOriginShape().GetDim(kCDimNCHWIdx);
  } else {
    dx_paras.co = out_backprop_desc->GetOriginShape().GetDim(kCDimNHWCIdx);
  }
  dx_paras.c1 = (dx_paras.cin + kBlockSize - 1) / kBlockSize;
  dx_paras.co1 = (dx_paras.co + kBlockSize - 1) / kBlockSize;
  dx_paras.h = y_desc->GetShape().GetDim(kHDimNC1HWC0Idx);
  dx_paras.w = y_desc->GetShape().GetDim(kWDimNC1HWC0Idx);
}

bool Conv2DBackpropInputParseFunc(const ge::OpDescPtr& op_desc, const nlohmann::json& compile_info,
                                  DxParas& dx_paras) {
  if (compile_info.contains("tiling_type") && compile_info["tiling_type"] == "binary") {
    dx_paras.repo_binary_flag = true;
    int32_t out_backprop_input_index = kInputIndexTwo;
    int32_t filter_input_index = 1;
    ge::OpDescPtr op_desc_attr = nullptr;
    GetInfo(dx_paras, compile_info);
    if (!GetAttrFromOp(op_desc, dx_paras)) {
      GELOGD("get attr from single op fail, try get attr from original fusion graph");
      UpdateOpDescAttr(op_desc, op_desc_attr);
      CHECK_OP_FUNC(!GetAttrFromOp(op_desc_attr, dx_paras), return false, "get attr failed");
    }
    bool stride_equal_one = dx_paras.stride_h == 1 && dx_paras.stride_w == 1;
    if (op_desc_attr != nullptr && stride_equal_one && dx_paras.binary_mode == KBinaryModeNCHW) {
      filter_input_index = kInputIndexTwo;
      out_backprop_input_index = 0;
    }
    ge::ConstGeTensorDescPtr filter_desc = op_desc->GetInputDescPtr(filter_input_index);
    ge::ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(out_backprop_input_index);
    ge::ConstGeTensorDescPtr y_desc = op_desc->GetOutputDescPtr(0);
    auto filter_format = filter_desc->GetFormat();
    auto y_format = y_desc->GetFormat();
    auto out_backprop_format = out_backprop_desc->GetFormat();
    auto y_dim_num = y_desc->GetShape().GetDimNum();
    auto out_backprop_dim_num = out_backprop_desc->GetShape().GetDimNum();
    auto y_ori_format = y_desc->GetOriginFormat();
    auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
    auto filter_ori_format = filter_desc->GetOriginFormat();
    uint32_t shift = 0;
    uint32_t parse_func_invalid = (!compile_info.contains("block_dim") << shift++);
    parse_func_invalid = parse_func_invalid + (!compile_info["block_dim"].contains("CORE_NUM") << shift++);
    parse_func_invalid = parse_func_invalid + ((filter_desc == nullptr) << shift++);
    parse_func_invalid = parse_func_invalid + ((out_backprop_desc == nullptr) << shift++);
    parse_func_invalid = parse_func_invalid + ((y_desc == nullptr) << shift++);
    if (dx_paras.binary_mode == KBinaryModeNC1HWC0) {
      parse_func_invalid =
          parse_func_invalid + ((y_ori_format != ge::FORMAT_NCHW && y_ori_format != ge::FORMAT_NHWC) << shift++);
      parse_func_invalid =
          parse_func_invalid +
          ((out_backprop_ori_format != ge::FORMAT_NCHW && out_backprop_ori_format != ge::FORMAT_NHWC) << shift++);
    } else {
      parse_func_invalid = parse_func_invalid + ((y_ori_format != ge::FORMAT_NCHW) << shift++);
      parse_func_invalid = parse_func_invalid + ((out_backprop_ori_format != ge::FORMAT_NCHW) << shift++);
    }
    parse_func_invalid = parse_func_invalid + ((filter_format != ge::FORMAT_FRACTAL_Z) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((filter_ori_format != ge::FORMAT_NCHW && filter_ori_format != ge::FORMAT_HWCN &&
                               filter_ori_format != ge::FORMAT_NHWC)
                              << shift++);
    parse_func_invalid =
        parse_func_invalid + ((y_format != ge::FORMAT_NCHW && y_format != ge::FORMAT_NC1HWC0) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((out_backprop_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((filter_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((filter_desc->GetShape().GetDimNum() != kConv2dNCHWSize) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((y_desc->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
    parse_func_invalid =
        parse_func_invalid + ((y_dim_num != kConv2dNCHWSize && y_dim_num != kConv2dNC1HWC0Size) << shift++);
    parse_func_invalid =
        parse_func_invalid +
        ((out_backprop_dim_num != kConv2dNCHWSize && out_backprop_dim_num != kConv2dNC1HWC0Size) << shift++);
    if (stride_equal_one) {
      dx_paras.stride_expand_flag = 0;
      parse_func_invalid =
          parse_func_invalid +
          ((out_backprop_format != ge::FORMAT_NCHW && out_backprop_format != ge::FORMAT_NC1HWC0) << shift++);
    } else {
      dx_paras.stride_expand_flag = 1;
      parse_func_invalid = parse_func_invalid + ((out_backprop_format != ge::FORMAT_NC1HWC0) << shift++);
    }
    if (parse_func_invalid != 0) {
      string error_info[shift++] = {"get block_dim failed", "get core_num failed", "tensor filter desc failed",
                                    "tensor out_backprop desc failed", "tensor y desc failed", "y ori_format invalid",
                                    "out_backprop ori_format failed", "filter format failed",
                                    "filter ori_format failed", "y format invalid",
                                    "out_backprop ori_shape len is invalid",
                                    "filter ori_shape len is invalid", "filter shape len is invalid",
                                    "y ori_shape len is invalid", "y shape len is invalid",
                                    "out_backprop shape len is invalid", "out_backprop format failed"};
      uint64_t error_num = static_cast<uint64_t>(parse_func_invalid);
      string error_flag = IntToBinary(error_num);
      OutputErrorMsg(error_info, error_flag);
      return false;
    }
    dx_paras.core_num = compile_info["block_dim"]["CORE_NUM"];
    CalShapeInfoFromDesc(dx_paras, filter_desc, out_backprop_desc, y_desc);
    CalPads(dx_paras);
    CalShapeInfo(dx_paras);
  }
  return true;
}

bool ConfigNoOverlapPara(DxParas &params) {
  // no_overlap is for non-32B aligned scenes to move out
  params.hw = params.h * params.w;
  params.dx_hw_align_flag = params.hw % kBlockSize == 0;
  params.dx_c_align_flag = params.cin % kBlockSize == 0;
  // The first case is that dx_hw is greater than 16, while non-16 alignment
  params.dx_no_overlap_condition_1 = !params.dx_hw_align_flag && (params.hw > kBlockSize);
  // The second case is that dx_hw is less than 16, dx_c is greater than 16 and dx_c is not 16 aligned
  params.dx_no_overlap_condition_2 = (params.hw < kBlockSize) && !params.dx_c_align_flag && params.cin > kBlockSize;
  return true;
}

void PretreatCompileInfo(const nlohmann::json &op_compile_info, nlohmann::json &op_info) {
  nlohmann::json item;
  for (size_t i = 1; i < op_compile_info.size(); ++i) {
    item = op_compile_info[i];
    std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
    for (auto &key : key_list) {
      auto &item_key = item[key];
      bool item_key_valid = item_key.is_object() && !item_key.empty();
      if (item_key_valid) {
        std::vector<int32_t> list_value = item_key.begin().value().get<std::vector<int32_t>>();
        op_info[key][item_key.begin().key()] = list_value;
      }
    }
    std::string key_int = "block_dim";
    auto &item_key_int = item[key_int];
    if (item_key_int.is_object() && !item_key_int.empty()) {
      int32_t int_value = item_key_int.begin().value().get<int32_t>();
      op_info[key_int][item_key_int.begin().key()] = int_value;
    }
  }
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
  try {
    DxParas dx_paras;
    CHECK_OP_FUNC(!Conv2DBackpropInputParseFunc(op_desc, opCompileInfo, dx_paras), return false, "ParseFunc failed!");
    if (dx_paras.repo_binary_flag) {
      int32_t tiling_id;
      Tiling tiling;
      bool cache_tiling_invalid = !CheckParams(dx_paras) || !ConfigNoOverlapPara(dx_paras) ||
                                  !GenTiling(dx_paras, tiling, tiling_id) ||
                                  !UpdateRunInfoBinary(dx_paras, tiling, tiling_id, runInfo);
      if (cache_tiling_invalid) {
        OP_LOGE("Conv2DBackpropInput", "binary mode failed");
        return false;
      }
      return true;
    }
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
    CHECK_OP_FUNC(opCompileInfo.empty(), return false, "op compile info is empty");
    // accurate build has only one item
    // fuzzy build has multiple items
    std::vector<std::string> varMap;
    nlohmann::json opInfo;
    GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
    if (opCompileInfo.is_array()) {
      // >>> start: splice compile info
      opInfo = opCompileInfo[0];
      varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();
      PretreatCompileInfo(opCompileInfo, opInfo);
      // <<< end: put together compile info
      GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
    } else if (opCompileInfo.is_object()) {
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

static bool GetAttrFromOp(gert::TilingContext *context, DxParas &dx_paras, bool depthwise) {
  auto attrs = context->GetAttrs();
  OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs"),
                  return false);

  const gert::ContinuousVector *strides_list = nullptr;
  const gert::ContinuousVector *pads_list = nullptr;
  const gert::ContinuousVector *dilations_list = nullptr;
  const int64_t *groups = nullptr;

  size_t idx = 0;
  strides_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  if (depthwise) {
    // DepthwiseConv2DBackpropInput attr: strides, dilations, pads, data_format
    dilations_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    pads_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  } else {
    // Conv2DBackpropInput attr: strides, pads, dilations, groups, data_format
    pads_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    dilations_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    groups = attrs->GetAttrPointer<int64_t>(idx++);
  }

  uint32_t shift = 0;
  uint32_t get_attr_success = ((strides_list == nullptr) << shift++);
  get_attr_success += ((strides_list->GetSize() != kConv2dDimSizeLimit) << shift++);
  get_attr_success += ((pads_list == nullptr) << shift++);
  get_attr_success += ((pads_list->GetSize() != kConv2dDimSizeLimit) << shift++);
  get_attr_success += ((dilations_list == nullptr) << shift++);
  get_attr_success += ((dilations_list->GetSize() != kConv2dDimSizeLimit) << shift++);
  if (!depthwise) {
    get_attr_success += ((groups == nullptr) << shift++);
  }
  if (get_attr_success != 0) {
    string error_info[shift++] = {
        "get strides failed",   "strides is invalid",   "get pads failed",   "pads is invalid",
        "get dilations failed", "dilations is invalid", "get groups failed",
    };

    uint64_t error_num = static_cast<uint64_t>(get_attr_success);
    string error_flag = IntToBinary(error_num);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }

  const int64_t *strides_list_data = reinterpret_cast<const int64_t *>(strides_list->GetData());
  const int64_t *pads_list_data = reinterpret_cast<const int64_t *>(pads_list->GetData());
  const int64_t *dilations_list_data = reinterpret_cast<const int64_t *>(dilations_list->GetData());

  dx_paras.groups = depthwise ? static_cast<int32_t>(*groups) : 1;
  auto y_ori_format = context->GetOutputDesc(0)->GetOriginFormat();
  if (y_ori_format == ge::FORMAT_NCHW) {
    dx_paras.stride_h = static_cast<int32_t>(strides_list_data[kHDimNCHWIdx]);
    dx_paras.stride_w = static_cast<int32_t>(strides_list_data[kWDimNCHWIdx]);
    dx_paras.dilations_n = static_cast<int32_t>(dilations_list_data[kNDimNCHWIdx]);
    dx_paras.dilations_c = static_cast<int32_t>(dilations_list_data[kCDimNCHWIdx]);
    dx_paras.dilations_h = static_cast<int32_t>(dilations_list_data[kHDimNCHWIdx]);
    dx_paras.dilations_w = static_cast<int32_t>(dilations_list_data[kWDimNCHWIdx]);
  } else {
    dx_paras.stride_h = static_cast<int32_t>(strides_list_data[kHDimNHWCIdx]);
    dx_paras.stride_w = static_cast<int32_t>(strides_list_data[kWDimNHWCIdx]);
    dx_paras.dilations_n = static_cast<int32_t>(dilations_list_data[kNDimNHWCIdx]);
    dx_paras.dilations_c = static_cast<int32_t>(dilations_list_data[kCDimNHWCIdx]);
    dx_paras.dilations_h = static_cast<int32_t>(dilations_list_data[kHDimNHWCIdx]);
    dx_paras.dilations_w = static_cast<int32_t>(dilations_list_data[kWDimNHWCIdx]);
  }

  dx_paras.padu = static_cast<int32_t>(pads_list_data[kConv2dPadUpIdx]);
  dx_paras.padd = static_cast<int32_t>(pads_list_data[kConv2dPadDownIdx]);
  dx_paras.padl = static_cast<int32_t>(pads_list_data[kConv2dPadLeftIdx]);
  dx_paras.padr = static_cast<int32_t>(pads_list_data[kConv2dPadRightIdx]);
  return true;
}

static void CalShapeInfoFromDesc(gert::TilingContext *context, size_t filter_input_index,
                                 size_t out_backprop_input_index, DxParas &dx_paras) {
  auto filter_desc = context->GetInputDesc(filter_input_index);
  auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  auto y_desc = context->GetOutputDesc(0);

  auto filter_shape = context->GetInputShape(filter_input_index);
  auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
  auto y_shape = context->GetOutputShape(0);

  dx_paras.batch = out_backprop_shape->GetStorageShape().GetDim(kNDimNC1HWC0Idx);
  dx_paras.batch_o = y_shape->GetStorageShape().GetDim(kNDimNC1HWC0Idx);
  dx_paras.ho = out_backprop_shape->GetStorageShape().GetDim(kHDimNC1HWC0Idx);
  dx_paras.wo = out_backprop_shape->GetStorageShape().GetDim(kWDimNC1HWC0Idx);
  dx_paras.filter_cin1hw = filter_shape->GetStorageShape().GetDim(0);
  dx_paras.filter_cout1 = filter_shape->GetStorageShape().GetDim(1);
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto &filter_ori_shape = filter_shape->GetOriginShape();
  if (filter_ori_format == ge::FORMAT_NCHW) {
    dx_paras.kn = filter_ori_shape.GetDim(kNDimNCHWIdx);
    dx_paras.kc = filter_ori_shape.GetDim(kCDimNCHWIdx);
    dx_paras.kh = filter_ori_shape.GetDim(kHDimNCHWIdx);
    dx_paras.kw = filter_ori_shape.GetDim(kWDimNCHWIdx);
  } else if (filter_ori_format == ge::FORMAT_HWCN) {
    dx_paras.kn = filter_ori_shape.GetDim(kNDimHWCNIdx);
    dx_paras.kc = filter_ori_shape.GetDim(kCDimHWCNIdx);
    dx_paras.kh = filter_ori_shape.GetDim(kHDimHWCNIdx);
    dx_paras.kw = filter_ori_shape.GetDim(kWDimHWCNIdx);
  } else {
    dx_paras.kn = filter_ori_shape.GetDim(kNDimNHWCIdx);
    dx_paras.kc = filter_ori_shape.GetDim(kCDimNHWCIdx);
    dx_paras.kh = filter_ori_shape.GetDim(kHDimNHWCIdx);
    dx_paras.kw = filter_ori_shape.GetDim(kWDimNHWCIdx);
  }
  if (y_desc->GetOriginFormat() == ge::FORMAT_NCHW) {
    dx_paras.cin = y_shape->GetOriginShape().GetDim(kCDimNCHWIdx);
  } else {
    dx_paras.cin = y_shape->GetOriginShape().GetDim(kCDimNHWCIdx);
  }
  if (out_backprop_desc->GetOriginFormat() == ge::FORMAT_NCHW) {
    dx_paras.co = out_backprop_shape->GetOriginShape().GetDim(kCDimNCHWIdx);
  } else {
    dx_paras.co = out_backprop_shape->GetOriginShape().GetDim(kCDimNHWCIdx);
  }
  dx_paras.c1 = (dx_paras.cin + kBlockSize - 1) / kBlockSize;
  dx_paras.co1 = (dx_paras.co + kBlockSize - 1) / kBlockSize;
  dx_paras.h = y_shape->GetStorageShape().GetDim(kHDimNC1HWC0Idx);
  dx_paras.w = y_shape->GetStorageShape().GetDim(kWDimNC1HWC0Idx);
}

static bool CalPadsAndGroups(gert::TilingContext *context, bool depthwise, DxParas &dx_paras) {
  if (dx_paras.kc == 0 || dx_paras.cin % dx_paras.kc != 0) {
    OP_LOGE(context->GetNodeName(), "fmap_channel(%d) %% filter_channel(%d) != 0", dx_paras.cin, dx_paras.kc);
    return false;
  }

  int32_t groups = dx_paras.cin / dx_paras.kc;
  if (dx_paras.groups == 1) {
    dx_paras.groups = groups;
  } else if (groups != dx_paras.groups) {
    OP_LOGE(context->GetNodeName(), "fmap_channel(%d) / filter_channel(%d) != groups(%d)", dx_paras.cin, dx_paras.kc,
            dx_paras.groups);
    return false;
  }

  int32_t filter_h = (dx_paras.kh - 1) * dx_paras.dilations_h + 1;
  int32_t filter_w = (dx_paras.kw - 1) * dx_paras.dilations_w + 1;
  size_t padding_attr_idx = depthwise ? 4 : 5;  // DepthwiseConv2DBackpropInput: 4, Conv2DBackpropInput: 5
  auto attrs = context->GetAttrs();
  if (attrs->GetAttrNum() <= padding_attr_idx) {
    OP_LOGD(context->GetNodeName(), "no padding attr, skip calc and check");
    return true;
  }
  auto padding = attrs->GetAttrPointer<char>(padding_attr_idx);
  if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
    int32_t pad_h = max(Align(dx_paras.h, dx_paras.stride_h) - dx_paras.stride_h + filter_h - dx_paras.h, 0);
    int32_t pad_up = (pad_h >> 1L);
    int32_t pad_down = pad_h - pad_up;
    int32_t pad_w = max(Align(dx_paras.w, dx_paras.stride_w) - dx_paras.stride_w + filter_w - dx_paras.w, 0);
    int32_t pad_left = (pad_w >> 1L);
    int32_t pad_right = pad_w - pad_left;
    dx_paras.padu = pad_up;
    dx_paras.padd = pad_down;
    dx_paras.padl = pad_left;
    dx_paras.padr = pad_right;
  }

  int32_t ho_expect = (dx_paras.h + dx_paras.padu + dx_paras.padd - filter_h) / dx_paras.stride_h + 1;
  int32_t wo_expect = (dx_paras.w + dx_paras.padl + dx_paras.padr - filter_w) / dx_paras.stride_w + 1;
  OP_TILING_CHECK(ho_expect != dx_paras.ho || wo_expect != dx_paras.wo,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                        "check pads attrs failed, ho: %d, wo: %d, ho_expect: %d, wo_expect: %d",
                                        dx_paras.ho, dx_paras.wo, ho_expect, wo_expect),
                  return false);
  return true;
}

static bool Conv2DBackpropInputParseFunc(gert::TilingContext *context, bool depthwise, DxParas &dx_paras) {
  size_t out_backprop_input_index = static_cast<size_t>(kInputIndexTwo);
  size_t filter_input_index = 1;
  if (!GetAttrFromOp(context, dx_paras, depthwise)) {
    OP_LOGE(context->GetNodeName(), "get attr from single op fail, try get attr from original fusion graph");
    return false;
  }

  bool stride_equal_one = dx_paras.stride_h == 1 && dx_paras.stride_w == 1;
  if (stride_equal_one && dx_paras.binary_mode == KBinaryModeNCHW) {
    filter_input_index = kInputIndexTwo;
    out_backprop_input_index = 0;
  }

  auto filter_desc = context->GetInputDesc(filter_input_index);
  auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  auto y_desc = context->GetOutputDesc(0);

  auto filter_shape = context->GetInputShape(filter_input_index);
  auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
  auto y_shape = context->GetOutputShape(0);

  auto filter_format = filter_desc->GetStorageFormat();
  auto y_format = y_desc->GetStorageFormat();
  auto out_backprop_format = out_backprop_desc->GetStorageFormat();
  auto y_dim_num = y_shape->GetStorageShape().GetDimNum();
  auto out_backprop_dim_num = out_backprop_shape->GetStorageShape().GetDimNum();
  auto y_ori_format = y_desc->GetOriginFormat();
  auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
  auto filter_ori_format = filter_desc->GetOriginFormat();
  uint64_t shift = 0;
  uint64_t error_num = ((filter_desc == nullptr || filter_shape == nullptr) << shift++);
  error_num += ((out_backprop_desc == nullptr || out_backprop_shape == nullptr) << shift++);
  error_num += ((y_desc == nullptr || y_shape == nullptr) << shift++);
  if (dx_paras.binary_mode == KBinaryModeNC1HWC0) {
    error_num += ((y_ori_format != FORMAT_NCHW && y_ori_format != FORMAT_NHWC) << shift++);
    error_num += ((out_backprop_ori_format != FORMAT_NCHW && out_backprop_ori_format != FORMAT_NHWC) << shift++);
  } else {
    error_num += ((y_ori_format != FORMAT_NCHW) << shift++);
    error_num += ((out_backprop_ori_format != FORMAT_NCHW) << shift++);
  }
  error_num += ((filter_format != FORMAT_FRACTAL_Z) << shift++);
  error_num +=
      ((filter_ori_format != FORMAT_NCHW && filter_ori_format != FORMAT_HWCN && filter_ori_format != FORMAT_NHWC)
       << shift++);
  error_num += ((y_format != FORMAT_NCHW && y_format != FORMAT_NC1HWC0) << shift++);
  error_num += ((out_backprop_shape->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
  error_num += ((filter_shape->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
  error_num += ((filter_shape->GetStorageShape().GetDimNum() != kConv2dNCHWSize) << shift++);
  error_num += ((y_shape->GetOriginShape().GetDimNum() != kConv2dNCHWSize) << shift++);
  error_num += ((y_dim_num != kConv2dNCHWSize && y_dim_num != kConv2dNC1HWC0Size) << shift++);
  error_num += ((out_backprop_dim_num != kConv2dNCHWSize && out_backprop_dim_num != kConv2dNC1HWC0Size) << shift++);
  if (stride_equal_one) {
    dx_paras.stride_expand_flag = 0;
    error_num += ((out_backprop_format != FORMAT_NCHW && out_backprop_format != FORMAT_NC1HWC0) << shift++);
  } else {
    dx_paras.stride_expand_flag = 1;
    error_num += ((out_backprop_format != FORMAT_NC1HWC0) << shift++);
  }
  if (error_num != 0) {
    string error_info[shift++] = {"tensor filter desc failed", "tensor out_backprop desc failed",
                                  "tensor y desc failed", "y ori_format invalid", "out_backprop ori_format failed",
                                  "filter format failed", "filter ori_format failed", "y format invalid",
                                  "out_backprop ori_shape len is invalid", "filter ori_shape len is invalid",
                                  "filter shape len is invalid", "y ori_shape len is invalid",
                                  "y shape len is invalid", "out_backprop shape len is invalid",
                                  "out_backprop format failed"};

    string error_flag = IntToBinary(error_num);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }

  CalShapeInfoFromDesc(context, filter_input_index, out_backprop_input_index, dx_paras);
  OP_TILING_CHECK(!CalPadsAndGroups(context, depthwise, dx_paras),
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to calc groups/pads"), return false);
  CalShapeInfo(dx_paras);
  return true;
}

static bool UpdateRunInfoBinary(const DxParas &params, const Tiling &tiling, int32_t tiling_id,
                                gert::TilingContext *context) {
  auto tiling_data = context->GetRawTilingData();
  size_t capacity = tiling_data->GetCapacity();
  bool stride_equal_one = params.stride_h == 1 && params.stride_w == 1;
  if (params.binary_mode == 1) {
    OP_TILING_CHECK(capacity < sizeof(RunInfoParaAubNoFusion),
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "insufficient tiling data capacity %zu", capacity),
                    return false);
    auto run = reinterpret_cast<RunInfoParaAubNoFusion *>(tiling_data->GetData());
    SetRunInfoAubNoFusion(params, tiling, *run);
    tiling_data->SetDataSize(sizeof(RunInfoParaAubNoFusion));
  } else if (params.binary_mode == kNumTwo) {
    if (stride_equal_one) {
      OP_TILING_CHECK(capacity < sizeof(RunInfoParaAubFusion),
                      CUBE_INNER_ERR_REPORT(context->GetNodeName(), "insufficient tiling data capacity %zu", capacity),
                      return false);
      auto run = reinterpret_cast<RunInfoParaAubFusion *>(tiling_data->GetData());
      SetRunInfoAubFusion(params, tiling, *run);
      tiling_data->SetDataSize(sizeof(RunInfoParaAubFusion));
    } else {
      OP_TILING_CHECK(capacity < sizeof(RunInfoParaAubNoFusion),
                      CUBE_INNER_ERR_REPORT(context->GetNodeName(), "insufficient tiling data capacity %zu", capacity),
                      return false);
      auto run = reinterpret_cast<RunInfoParaAubNoFusion *>(tiling_data->GetData());
      SetRunInfoAubNoFusion(params, tiling, *run);
      tiling_data->SetDataSize(sizeof(RunInfoParaAubNoFusion));
    }
  }

  context->SetBlockDim(static_cast<uint32_t>(tiling.batch_dim * tiling.n_dim * tiling.m_dim));
  context->SetTilingKey(static_cast<uint64_t>(tiling_id));
  return true;
}

static size_t InitVarsValues(uint32_t var_bit_flags, const gert::Shape &in_shape, const gert::Shape &out_shape,
                             gert::Shape &var_value, int64_t *shape_for_range_match) {
  if ((var_bit_flags & kVarBatchN) != 0) {
    var_value.AppendDim(out_shape.GetDim(kConv2dNDim));
  }

  if ((var_bit_flags & kVarDxH) != 0) {
    var_value.AppendDim(in_shape.GetDim(kConv2dHDim));
    var_value.AppendDim(out_shape.GetDim(kConv2dHDim));
  }

  if ((var_bit_flags & kVarDxW) != 0) {
    var_value.AppendDim(in_shape.GetDim(kConv2dWDim));
    var_value.AppendDim(out_shape.GetDim(kConv2dWDim));
  }

  size_t idx = 0;
  shape_for_range_match[idx++] = out_shape.GetDim(kConv2dNDim);
  if (var_value.GetDimNum() > 1) {  // not only dynamic batch
    shape_for_range_match[idx++] = out_shape.GetDim(kConv2dHDim);
    shape_for_range_match[idx++] = out_shape.GetDim(kConv2dWDim);
  }

  return idx;
}

ge::graphStatus TilingForConv2DDx(gert::TilingContext *context, bool depthwise) {
  auto compile_info = reinterpret_cast<const Conv2DBackPropCompileInfo *>(context->GetCompileInfo());
  OP_TILING_CHECK(compile_info == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compile_info is null"),
                  return ge::GRAPH_FAILED);

  if (compile_info->repo_binary_flag) {
    DxParas dx_paras;
    OP_TILING_CHECK(!Conv2DBackpropInputParseFunc(context, depthwise, dx_paras),
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to parse context"), return ge::GRAPH_FAILED);
    dx_paras.core_num = compile_info->core_num;
    dx_paras.aub_num = compile_info->aub_num;
    dx_paras.cub_num = compile_info->cub_num;
    dx_paras.binary_mode = compile_info->binary_mode;
    dx_paras.ub_size = compile_info->ub_size;

    int32_t tiling_id;
    Tiling tiling;
    bool cache_tiling_invalid = !CheckParams(dx_paras) || !ConfigNoOverlapPara(dx_paras) ||
                                !GenTiling(dx_paras, tiling, tiling_id) ||
                                !UpdateRunInfoBinary(dx_paras, tiling, tiling_id, context);
    if (cache_tiling_invalid) {
      OP_LOGE(context->GetNodeName(), "binary mode failed");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  auto tensor_in_desc = context->GetInputDesc(2);  // 2: out_backprop
  auto tensor_out_desc = context->GetOutputDesc(0);
  auto tensor_in_shape = context->GetInputShape(2);  // 2: out_backprop
  auto tensor_out_shape = context->GetOutputShape(0);

  OP_TILING_CHECK(tensor_in_desc == nullptr || tensor_out_desc == nullptr || tensor_in_shape == nullptr ||
                      tensor_out_shape == nullptr,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input/output shape/tensor"),
                  return ge::GRAPH_FAILED);

  const auto &in_shape = tensor_in_shape->GetStorageShape();
  const auto &out_shape = tensor_out_shape->GetStorageShape();
  bool unvalid_size = context->GetComputeNodeInfo()->GetInputsNum() < kConv2dDxInputSizeLimit ||
                      context->GetComputeNodeInfo()->GetOutputsNum() == 0 ||
                      in_shape.GetDimNum() < kConv2dDimNumLimit || out_shape.GetDimNum() < kConv2dDimNumLimit;
  OP_TILING_CHECK(unvalid_size, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "the size is unvalid."),
                  return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "Current format is %s, Ori format is %s",
          ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetStorageFormat()).c_str(),
          ge::TypeUtils::FormatToSerialString(tensor_out_desc->GetOriginFormat()).c_str());

  gert::Shape var_value;
  int64_t shape_for_range_match[3];  // 3: nhw
  size_t dim_num = InitVarsValues(compile_info->var_bit_flags, in_shape, out_shape, var_value, shape_for_range_match);
  return CubeTiling(shape_for_range_match, dim_num, var_value, *compile_info, context);
}

ge::graphStatus TilingForConv2DBpInput(gert::TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("Conv2DBackpropInput", "context is null"),
                  return ge::GRAPH_FAILED);
  return TilingForConv2DDx(context, false);
}

ge::graphStatus TilingForDepthwiseConv2DBackpropInput(gert::TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("DepthwiseConv2DBackpropInput", "context is null"),
                  return ge::GRAPH_FAILED);
  return TilingForConv2DDx(context, true);
}

IMPL_OP(Conv2DBackpropInput)
    .Tiling(TilingForConv2DBpInput)
    .TilingParse<Conv2DBackPropCompileInfo>(ParseConv2DBackpropCompileInfo);

IMPL_OP(DepthwiseConv2DBackpropInput)
    .Tiling(TilingForDepthwiseConv2DBackpropInput)
    .TilingParse<Conv2DBackPropCompileInfo>(ParseConv2DBackpropCompileInfo);
}  // namespace optiling
