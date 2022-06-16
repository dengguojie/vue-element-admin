/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv2d_backprop_filter.cpp
 * \brief tiling function of conv2d_backprop_filter
 */
#include <vector>
#include <string>

#include "cube_tiling_new.h"
#include "cube_tiling_runtime.h"
#include "dw_cache_tiling.h"
#include "op_tiling.h"
#include "op_log.h"
#include "graph/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "external/graph/operator.h"
#include "exe_graph/runtime/tiling_context.h"
#include "error_log.h"
#include "register/op_impl_registry.h"

#include <nlohmann/json.hpp>

using namespace std;
namespace optiling {
const size_t kConv2dDwInputSizeLimit = 3;
// pads idx
const int32_t kPadsUpDim = 0;
const int32_t kPadsDownDim = 1;
const int32_t kPadsLeftDim = 2;
const int32_t kPadsRightDim = 3;

const size_t kPos0 = 0;
const size_t kPos1 = 1;
const size_t kPos2 = 2;
const size_t kPos3 = 3;

const size_t kNchwDimN = 0;
const size_t kNchwDimC = 1;
const size_t kNchwDimH = 2;
const size_t kNchwDimW = 3;

const int32_t kStrideDim = 4;
const int32_t kPadsDim = 4;
const int32_t kDilationDim = 4;
const size_t kOriShapeDim = 4;

const int64_t kDimLow = 1;
const int64_t kDimUpper = INT_MAX;
const int64_t kFilterHWUpper = 255;
const int64_t kStrideHWUpper = 63;
const int64_t kDimHWLow = 2;
const int64_t kDimHWUpper = 4096;
const int64_t kDefaultC0 = 16;

const int32_t kBlockDimRate = 2;
const int32_t kFloat16Bytes = 2;

const int64_t kDefaultL1Size = (1024 * 1024);

template <typename T>
inline T Align(T param1, T param2) {
  return (param1 + param2 - 1) / param2 * param2;
}

inline bool CheckRange(int64_t val, int64_t down, int64_t upper) {
  return val >= down && val <= upper;
}

bool UpdateRunInfoCube(const conv2d_dw::Conv2dDwTiling& tiling, const conv2d_dw::Conv2dBpFilterParas& params,
                       optiling::utils::OpRunInfo& run_info) {
  uint32_t block_dim = tiling.batch_dim * tiling.n_dim * tiling.h_dim * tiling.m_dim * tiling.group_dim;
  run_info.SetBlockDim(block_dim);
  run_info.SetTilingKey(stoi(tiling.tiling_id));
  // transdata vars
  run_info.AddTilingData(params.batch);
  run_info.AddTilingData(params.ci);
  run_info.AddTilingData(params.hi);
  run_info.AddTilingData(params.wi);
  run_info.AddTilingData(params.co);
  run_info.AddTilingData(params.ho);
  run_info.AddTilingData(params.wo);
  run_info.AddTilingData(params.kh);
  run_info.AddTilingData(params.kw);
  // dw vars
  run_info.AddTilingData(params.ci1);
  run_info.AddTilingData(params.co1);
  run_info.AddTilingData(params.stride_h);
  run_info.AddTilingData(params.stride_w);
  run_info.AddTilingData(params.pad_u);
  run_info.AddTilingData(params.pad_d);
  run_info.AddTilingData(params.pad_l);
  run_info.AddTilingData(params.pad_r);
  run_info.AddTilingData(params.dilation_h);
  run_info.AddTilingData(params.dilation_w);
  run_info.AddTilingData(params.groups);
  // set tiling
  OP_LOGE_IF(params.ci0 == 0, false, params.op_type, "There is a div zero error. params:ci0 is zero.");
  OP_LOGE_IF(tiling.h_dim == 0, false, params.op_type, "There is a div zero error. tiling:h_dim is zero.");
  int32_t total_n = params.ci1 * params.kh * params.kw;
  int32_t single_core_k = ((params.ho * params.wo + params.ci0 - 1) / params.ci0) / tiling.h_dim;
  // group_dim
  run_info.AddTilingData(tiling.group_dim);
  // batch_dim
  run_info.AddTilingData(tiling.batch_dim);
  // k_dim
  run_info.AddTilingData(tiling.h_dim);
  // batch_single_core
  OP_LOGE_IF(tiling.batch_dim == 0, false, params.op_type, "There is a div zero error. tiling:batch_dim is zero.");
  run_info.AddTilingData(params.batch / tiling.batch_dim);
  // n_single_core
  OP_LOGE_IF(tiling.n_dim * tiling.n_bl1 * tiling.n_l0 == 0, false, params.op_type,
             "There is a div zero error. tiling:n_dim * n_bl1 * n_l0 is zero.");
  run_info.AddTilingData(total_n / (tiling.n_dim * tiling.n_bl1 * tiling.n_l0));
  // n_dim
  run_info.AddTilingData(tiling.n_dim);
  // n_bl1
  run_info.AddTilingData(tiling.n_bl1);
  // n_ub_l0_time = l0c_n / cub_n
  OP_LOGE_IF(tiling.n_cub == 0, false, params.op_type, "There is a div zero error. tiling:n_cub is zero.");
  run_info.AddTilingData(tiling.n_l0 / tiling.n_cub);
  // cub_n1
  run_info.AddTilingData(tiling.n_cub);
  // m_dim
  run_info.AddTilingData(tiling.m_dim);
  // m_single_core
  OP_LOGE_IF(tiling.m_dim * tiling.m_al1 * tiling.m_l0 == 0, false, params.op_type,
             "There is a div zero error. tiling:tiling.m_dim * tiling.m_al1 * tiling.m_l0 is zero.");
  run_info.AddTilingData(params.co1 / tiling.m_dim / tiling.m_al1 / tiling.m_l0);
  // m_al1
  run_info.AddTilingData(tiling.m_al1);
  // m_l0
  run_info.AddTilingData(tiling.m_l0);
  // k_l0
  run_info.AddTilingData(tiling.k_l0);
  // k_al1_factor
  OP_LOGE_IF(tiling.kal1_16 == 0, false, params.op_type, "There is a div zero error. tiling:kal1_16 is zero.");
  run_info.AddTilingData(single_core_k / tiling.kal1_16);
  // k_bl1_factor
  OP_LOGE_IF(tiling.kbl1_16 == 0, false, params.op_type, "There is a div zero error. tiling:kbl1_16 is zero.");
  run_info.AddTilingData(single_core_k / tiling.kbl1_16);
  // k_al0_factor
  OP_LOGE_IF(tiling.k_l0 == 0, false, params.op_type, "There is a div zero error. tiling:k_l0 is zero.");
  run_info.AddTilingData(tiling.kal1_16 / tiling.k_l0);
  // k_bl0_factor
  run_info.AddTilingData(tiling.kbl1_16 / tiling.k_l0);
  // k_al1_16
  run_info.AddTilingData(tiling.kal1_16);
  // k_bl1_16
  run_info.AddTilingData(tiling.kbl1_16);
  // kl1_times
  run_info.AddTilingData(std::max(tiling.kal1_16, tiling.kbl1_16) / std::min(tiling.kal1_16, tiling.kbl1_16));
  // bl1_bound
  run_info.AddTilingData(tiling.bl1_bound);
  // m_aub
  run_info.AddTilingData(tiling.m_aub);
  // n_bub
  run_info.AddTilingData(tiling.n_bub);
  // k_aub
  run_info.AddTilingData(tiling.k_aub);
  // k_bub
  run_info.AddTilingData(tiling.k_bub);
  // ho_bl1
  run_info.AddTilingData(tiling.ho_bl1);
  // multi_n_ub_l1
  OP_LOGE_IF(tiling.n_bub == 0, false, params.op_type, "There is a div zero error. tiling:n_bub is zero.");
  run_info.AddTilingData(tiling.n_bl1 * tiling.n_l0 / tiling.n_bub);
  // multi_m_ub_l1
  OP_LOGE_IF(tiling.m_aub == 0, false, params.op_type, "There is a div zero error. tiling:m_aub is zero.");
  run_info.AddTilingData(tiling.m_al1 * tiling.m_l0 / tiling.m_aub);
  // multi_k_aub_l1
  OP_LOGE_IF(tiling.k_aub == 0, false, params.op_type, "There is a div zero error. tiling:k_aub is zero.");
  run_info.AddTilingData(tiling.kal1_16 / tiling.k_aub);
  // multi_k_bub_l1
  OP_LOGE_IF(tiling.k_bub == 0, false, params.op_type, "There is a div zero error. tiling:k_bub is zero.");
  run_info.AddTilingData(tiling.kbl1_16 / tiling.k_bub);
  return true;
}

bool TransformShape(const ge::ConstGeTensorDescPtr& tensor, std::vector<int64_t>& shape) {
  const ge::Format ori_format = tensor->GetOriginFormat();
  const ge::GeShape& ori_shape = tensor->GetOriginShape();
  // check ori_shape dim
  OP_LOGE_IF(ori_shape.GetDimNum() != kOriShapeDim, false, "TransformShape", "ori shape dim nums is invalid.");

  if (ori_format == ge::FORMAT_NCHW) {
    shape = {ori_shape.GetDim(kPos0), ori_shape.GetDim(kPos1), ori_shape.GetDim(kPos2), ori_shape.GetDim(kPos3)};
  } else if (ori_format == ge::FORMAT_HWCN) {
    shape = {ori_shape.GetDim(kPos3), ori_shape.GetDim(kPos2), ori_shape.GetDim(kPos0), ori_shape.GetDim(kPos1)};
  } else if (ori_format == ge::FORMAT_NHWC) {
    shape = {ori_shape.GetDim(kPos0), ori_shape.GetDim(kPos3), ori_shape.GetDim(kPos1), ori_shape.GetDim(kPos2)};
  } else {
    return false;
  }
  return true;
}

void ModifyPad(conv2d_dw::Conv2dBpFilterParas& params) {
  GELOGD("before update pads, the pad_u is %d, pad_d is %d, pad_l is %d, pad_r is %d",
         params.pad_u, params.pad_d, params.pad_l, params.pad_r);
  int64_t filter_dilation_h = (params.kh - 1) * params.dilation_h + 1;
  int64_t filter_dilation_w = (params.kw - 1) * params.dilation_w + 1;
  int64_t pad_h = std::max(
      (int64_t)(Align(params.hi, params.stride_h) - params.stride_h + filter_dilation_h - params.hi),
      0L);
  int64_t pad_w = std::max(
      (int64_t)(Align(params.wi, params.stride_w) - params.stride_w + filter_dilation_w - params.wi),
      0L);
  params.pad_u = (pad_h >> 1L);
  params.pad_d = pad_h - params.pad_u;
  params.pad_l = (pad_w >> 1L);
  params.pad_r = pad_w - params.pad_l;
  GELOGD("after update pads, the pad_u is %d, pad_d is %d, pad_l is %d, pad_r is %d",
         params.pad_u, params.pad_d, params.pad_l, params.pad_r);
}

bool SetCacheTilingParamsFromOpDesc(const ge::OpDescPtr& op_desc, conv2d_dw::Conv2dBpFilterParas& params) {
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  std::string data_format;
  OP_LOGE_IF(!ge::AttrUtils::GetListInt(op_desc, "strides", strides), false, params.op_type,
             "get strides from op desc fail.");
  OP_LOGE_IF(strides.size() != kStrideDim, false, params.op_type, "strides of op desc dim len is invalid.");
  OP_LOGE_IF(strides[kNchwDimN] != 1 || strides[kNchwDimC] != 1, false, params.op_type, "strides N/C dim must be 1.");
  OP_LOGE_IF(!ge::AttrUtils::GetInt(op_desc, "groups", params.groups), false, params.op_type,
             "get strides from op desc fail.");
  OP_LOGE_IF(!ge::AttrUtils::GetListInt(op_desc, "dilations", dilations), false, params.op_type,
             "get dilations from op desc fail.");
  OP_LOGE_IF(dilations.size() != kDilationDim, false, params.op_type, "dilations of op desc dim len is invalid.");
  OP_LOGE_IF(dilations[kNchwDimN] != 1 || dilations[kNchwDimC] != 1, false, params.op_type,
             "dilations N/C dim must be 1.");
  OP_LOGE_IF(!ge::AttrUtils::GetListInt(op_desc, "pads", pads), false, params.op_type, "get pads from op desc fail.");
  OP_LOGE_IF(pads.size() != kPadsDim, false, params.op_type, "pads of op desc dim len is invalid.");
  OP_LOGE_IF(!ge::AttrUtils::GetStr(op_desc, "data_format", data_format), false, params.op_type,
             "get data_format from op desc fail.");
  OP_LOGE_IF(data_format != "NCHW", false, params.op_type, "current dw binary only support NCHW format.");
  params.pad_u = pads[kPadsUpDim];
  params.pad_d = pads[kPadsDownDim];
  params.pad_l = pads[kPadsLeftDim];
  params.pad_r = pads[kPadsRightDim];
  params.stride_h = strides[kNchwDimH];
  params.stride_w = strides[kNchwDimW];
  params.dilation_h = dilations[kNchwDimH];
  params.dilation_w = dilations[kNchwDimW];
  return true;
}

bool CheckShapeRelation(const conv2d_dw::Conv2dBpFilterParas& params) {
  OP_LOGE_IF((params.kh - 1) * params.dilation_h + 1 > params.hi + params.pad_u + params.pad_d, false, params.op_type,
             "kh value is invalid.");
  OP_LOGE_IF((params.kw - 1) * params.dilation_w + 1 > params.wi + params.pad_l + params.pad_r, false, params.op_type,
             "kh value is invalid.");
  OP_LOGE_IF((params.hi - params.kh + params.pad_u + params.pad_d) / params.stride_h + 1 != params.ho, false,
             params.op_type, "ho is not match with hi.");
  OP_LOGE_IF((params.wi - params.kw + params.pad_l + params.pad_r) / params.stride_w + 1 != params.wo, false,
             params.op_type, "wo is not match with wi.");

  return true;
}

bool CheckL1Size(const conv2d_dw::Conv2dBpFilterParas& params) {
  int32_t kernel_dilation_h = (params.kh - 1) * params.dilation_h + 1;
  int32_t al1_min_size = params.co0 * params.k0 * params.a_dtype * kFloat16Bytes;
  int32_t bl1_align_factor = (params.ci0 + params.wo - 1) / params.wo;
  if (params.ci0 % params.wo != 0 || params.wo % params.ci0 != 0) {
    ++bl1_align_factor;
  }
  int32_t bl1_min_size =
      Align<int32_t>(kernel_dilation_h + (bl1_align_factor - 1) * params.stride_h * params.wi, params.ci0) *
      params.ci0 * params.b_dtype * kFloat16Bytes;
  OP_LOGE_IF(al1_min_size + bl1_min_size > kDefaultL1Size, false, params.op_type, "L1 min tiling excced L1 size.");
  return true;
}

bool CheckParamsRange(const conv2d_dw::Conv2dBpFilterParas& params) {
  OP_LOGE_IF(!CheckRange(params.kh, kDimLow, kFilterHWUpper), false, params.op_type, "kh value is invalid.");
  OP_LOGE_IF(!CheckRange(params.kw, kDimLow, kFilterHWUpper), false, params.op_type, "kw value is invalid.");
  OP_LOGE_IF(!CheckRange(params.batch, kDimLow, kDimUpper), false, params.op_type, "batch value is invalid.");
  OP_LOGE_IF(!CheckRange(params.co1, kDimLow, kDimUpper), false, params.op_type, "co1 value is invalid.");
  OP_LOGE_IF(!CheckRange(params.ho, kDimHWLow, kDimHWUpper), false, params.op_type, "ho value is invalid.");
  OP_LOGE_IF(!CheckRange(params.wo, kDimHWLow, kDimHWUpper), false, params.op_type, "wo value is invalid.");
  OP_LOGE_IF(!CheckRange(params.co, kDimLow, kDimUpper), false, params.op_type, "co value is invalid.");
  OP_LOGE_IF(!CheckRange(params.ci1, kDimLow, kDimUpper), false, params.op_type, "c1 value is invalid.");
  OP_LOGE_IF(!CheckRange(params.ci, kDimLow, kDimUpper), false, params.op_type, "cin value is invalid.");
  OP_LOGE_IF(!CheckRange(params.groups, kDimLow, kDimLow), false, params.op_type, "groups only support 1 now.");
  OP_LOGE_IF(!CheckRange(params.ho * params.stride_h, kDimHWLow, kDimHWUpper), false, params.op_type,
             "ho * stride_h exceed supported range.");
  OP_LOGE_IF(!CheckRange(params.wo * params.stride_w, kDimHWLow, kDimHWUpper), false, params.op_type,
             "wo * stride_w exceed supported range.");
  OP_LOGE_IF(!CheckRange(params.hi, kDimHWLow, kDimHWUpper), false, params.op_type, "hi value is invalid.");
  OP_LOGE_IF(!CheckRange(params.wi, kDimHWLow, kDimHWUpper), false, params.op_type, "wi value is invalid.");
  OP_LOGE_IF(!CheckRange(params.stride_h, kDimLow, kStrideHWUpper), false, params.op_type,
             "stride_h value is invalid.");
  OP_LOGE_IF(!CheckRange(params.stride_w, kDimLow, kStrideHWUpper), false, params.op_type,
             "stride_w value is invalid.");

  OP_LOGE_IF(!CheckRange(params.dilation_h, kDimLow, kDimLow), false, params.op_type, "dilation_h value is invalid.");
  OP_LOGE_IF(!CheckRange(params.dilation_w, kDimLow, kDimLow), false, params.op_type, "dilation_w value is invalid.");
  OP_LOGE_IF(!CheckRange(params.ci0, kDefaultC0, kDefaultC0), false, params.op_type, "ci0 value is invalid.");
  OP_LOGE_IF(!CheckRange(params.co0, kDefaultC0, kDefaultC0), false, params.op_type, "co0 value is invalid.");
  return true;
}

bool CheckCacheTilingParams(const conv2d_dw::Conv2dBpFilterParas& params) {
  // check dtype
  OP_LOGE_IF(!CheckParamsRange(params), false, params.op_type, "Check dw cache tiling params value range failed.");
  // check shape,attrs rule
  OP_LOGE_IF(!CheckShapeRelation(params), false, params.op_type, "Check dw cache tiling params value range failed.");
  // check size exceed 64
  OP_LOGE_IF(!CheckL1Size(params), false, params.op_type, "dw cache tiling min l1 used size would exceed L1 soc size.");
  return true;
}

bool SetShapeParams(const ge::OpDescPtr& op_desc, conv2d_dw::Conv2dBpFilterParas& params,
                    const int32_t dedy_index) {
  ge::ConstGeTensorDescPtr dedy_desc = op_desc->GetInputDescPtr(dedy_index);
  ge::ConstGeTensorDescPtr filter_desc = op_desc->GetOutputDescPtr(0);
  ge::ConstGeTensorDescPtr dedx_desc = op_desc->GetInputDescPtr(0);

  std::vector<int64_t> dedy_shape_nchw;
  std::vector<int64_t> filter_shape_nchw;
  std::vector<int64_t> dedx_shape_nchw;
  OP_LOGE_IF(!TransformShape(dedy_desc, dedy_shape_nchw), false, params.op_type, "Transform dedy shape fail.");
  OP_LOGE_IF(!TransformShape(dedx_desc, dedx_shape_nchw), false, params.op_type, "Transform dedx shape fail.");
  OP_LOGE_IF(!TransformShape(filter_desc, filter_shape_nchw), false, params.op_type, "Transform filter shape fail.");

  params.batch = dedy_shape_nchw[kNchwDimN];
  params.ho = dedy_shape_nchw[kNchwDimH];
  params.wo = dedy_shape_nchw[kNchwDimW];
  params.co = dedy_shape_nchw[kNchwDimC];
  params.co0 = kDefaultC0;
  params.co1 = (params.co + params.co0 - 1) / params.co0;

  params.hi = dedx_shape_nchw[kNchwDimH];
  params.wi = dedx_shape_nchw[kNchwDimW];
  params.ci = dedx_shape_nchw[kNchwDimC];
  params.ci0 = kDefaultC0;
  params.ci1 = (params.ci + params.ci0 - 1) / params.ci0;

  params.kh = filter_shape_nchw[kNchwDimH];
  params.kw = filter_shape_nchw[kNchwDimW];
  if (params.pad_u == -1 || params.pad_l == -1) {
    ModifyPad(params);
  }
  return true;
}

bool ParseOpInfo(const nlohmann::json& compile_info, const ge::OpDescPtr& op_desc,
                 conv2d_dw::Conv2dBpFilterParas& params, const ge::OpDescPtr ori_dw_desc) {
  // get attrs from op_desc
  int32_t dedy_index = 2;
  ge::OpDescPtr op_dw_desc = nullptr;
  if (ori_dw_desc != nullptr) {
    op_dw_desc = ori_dw_desc;
    dedy_index = 1;
  } else {
    op_dw_desc = op_desc;
  }
  OP_LOGE_IF(!SetCacheTilingParamsFromOpDesc(op_dw_desc, params), false, params.op_type,
             "Set cache tiling params failed.");
  OP_LOGE_IF(!SetShapeParams(op_desc, params, dedy_index), false, params.op_type, "Set shape params failed.");
  OP_LOGE_IF(!compile_info.contains("max_core_num"), false, params.op_type,
             "compile info attrs not contains max_core_num.");
  // dw tiling can use double times core num
  params.max_core_num = static_cast<int>(compile_info["max_core_num"]) * kBlockDimRate;
  // check shape and attrs by params
  return true;
}

bool BinaryTiling(const string& op_type, const ge::OpDescPtr op_desc, const nlohmann::json& compile_info,
                  utils::OpRunInfo& run_info, const ge::OpDescPtr ori_dw_desc) {
  conv2d_dw::Conv2dBpFilterParas params;
  params.op_type = op_type;
  OP_LOGE_IF(!ParseOpInfo(compile_info, op_desc, params, ori_dw_desc),
             false, op_type, "Parse cache tiling params failed.");
  OP_LOGE_IF(!CheckCacheTilingParams(params), false, op_type, "Check cache tiling params failed.");
  conv2d_dw::Conv2dDwCacheTiling cacheTiling(params);
  conv2d_dw::Conv2dDwTiling tiling;
  OP_LOGE_IF(!cacheTiling.GenTiling(tiling), false, op_type, "GenTiling failed!");
  UpdateRunInfoCube(tiling, params, run_info);
  return true;
}

void SpliceCompileInfo(const nlohmann::json& op_compile_info, nlohmann::json& op_info) {
  nlohmann::json item;
  for (size_t i = 1; i < op_compile_info.size(); ++i) {
    item = op_compile_info[i];
    std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
    for (auto& key : key_list) {
      auto& item_key = item[key];
      if (item_key.is_object() && !item_key.empty()) {
        std::vector<int32_t> list_value = item_key.begin().value().get<std::vector<int32_t>>();
        op_info[key][item_key.begin().key()] = list_value;
      }
    }
    std::string key_int = "block_dim";
    auto& item_key_int = item[key_int];
    if (item_key_int.is_object() && !item_key_int.empty()) {
      int32_t int_value = item_key_int.begin().value().get<int32_t>();
      op_info[key_int][item_key_int.begin().key()] = int_value;
    }
  }
  // <<< end: put together compile info
  GELOGD("compile info after splice is: %s", op_info.dump().c_str());
}

/*
 * @brief: tiling function of conv2d_backprop_filter
 * @param [in] op_type: op_type of the conv2d_backprop_filter
 * @param [in] op_paras: inputs/outputs/atts of the conv2d_backprop_filter
 * @param [in] op_compile_info: compile time generated info of the conv2d_backprop_filter
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DBpFilterTiling(const std::string& op_type, const ge::Operator& op_paras,
                          const nlohmann::json& op_compile_info, utils::OpRunInfo& run_info) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  ge::ComputeGraphPtr ori_graph = nullptr;
  ge::OpDescPtr ori_dw_desc = nullptr;
  int32_t dedy_index = 2;
  if (ge::AttrUtils::GetGraph(op_desc, "_original_fusion_graph", ori_graph)) {
    dedy_index = 1;
    for (auto node : ori_graph->GetAllNodes()) {
      if (node->GetType() == "Conv2DBackpropFilter") {
        ori_dw_desc = node->GetOpDesc();
        break;
      }
    }
  } else {
    GELOGD("this is not fusion node, only conv2d_backprop_filter single node");
  }
  OP_LOGE_IF(op_desc == nullptr, false, op_type, "the op_desc is nullptr.");
  ge::ConstGeTensorDescPtr tensor_a_desc = op_desc->GetInputDescPtr(0);
  // the tensor b's index is 2
  ge::ConstGeTensorDescPtr tensor_b_desc = op_desc->GetInputDescPtr(dedy_index);
  const ge::GeShape& tensor_a_shape = tensor_a_desc->GetShape();
  const ge::GeShape& tensor_b_shape = tensor_b_desc->GetShape();
  size_t shape_a_dimnum = tensor_a_shape.GetDimNum();
  bool unvalid_size = op_paras.GetInputsSize() < kConv2dDwInputSizeLimit || op_paras.GetOutputsSize() == 0 ||
                      shape_a_dimnum < kConv2dDimNumLimit || tensor_b_shape.GetDimNum() < kConv2dDimNumLimit;
  OP_LOGE_IF(unvalid_size, false, op_type, "the size is unvalid.");
  GELOGD("Current format is %s, Ori format is %s",
         ge::TypeUtils::FormatToSerialString(tensor_a_desc->GetFormat()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_a_desc->GetOriginFormat()).c_str());

  try {
    OP_LOGE_IF(op_compile_info.empty(), false, op_type, "op compile info is empty.");
    // accurate build has only one item, fuzzy build has multiple items
    std::vector<std::string> var_map;
    nlohmann::json op_info;
    GELOGD("original compile info is: %s", op_compile_info.dump().c_str());
    if (op_compile_info.is_array()) {
      // >>> start: splice compile info
      op_info = op_compile_info[0];
      var_map = op_info.at("_vars").begin().value().get<std::vector<std::string>>();
      SpliceCompileInfo(op_compile_info, op_info);
    } else if (op_compile_info.is_object()) {
      if (op_compile_info.contains("tiling_type") && op_compile_info.at("tiling_type") == "binary") {
        BinaryTiling(op_type, op_desc, op_compile_info, run_info, ori_dw_desc);
        return true;
      }
      var_map = op_compile_info.at("_vars")["10000"].get<std::vector<std::string>>();
      op_info = op_compile_info;
    }

    std::vector<int64_t> var_value;
    if (std::find(var_map.begin(), var_map.end(), "batch") != var_map.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dNDim));
    }
    if (std::find(var_map.begin(), var_map.end(), "fmap_h") != var_map.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dHDim));
      var_value.insert(var_value.end(), tensor_b_shape.GetDim(kConv2dHDim));
    }
    if (std::find(var_map.begin(), var_map.end(), "fmap_w") != var_map.end()) {
      var_value.insert(var_value.end(), tensor_a_shape.GetDim(kConv2dWDim));
      var_value.insert(var_value.end(), tensor_b_shape.GetDim(kConv2dWDim));
    }

    std::vector<int64_t> input_shape;
    input_shape.reserve(shape_a_dimnum);
    for (size_t i = 0; i < shape_a_dimnum; i++) {
      input_shape.emplace_back(tensor_a_shape.GetDim(i));
    }
    return cube_tiling(op_type, input_shape, var_value, op_info, run_info);
  } catch (...) {
    GELOGD("get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the conv2d_backprop_filter
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv2DBackpropFilter, Conv2DBpFilterTiling);
REGISTER_OP_TILING_FUNC_BUFFERED_V2(DepthwiseConv2DBackpropFilter, Conv2DBpFilterTiling);

static bool SetCacheTilingParamsFromOpDesc(gert::TilingContext *context, conv2d_dw::Conv2dBpFilterParas &params,
                                           bool depthwise) {
  const auto op_name = context->GetNodeName();
  const auto attrs = context->GetAttrs();
  OP_LOGE_IF(attrs == nullptr, false, op_name, "failed to get attrs from context.");

  const gert::ContinuousVector *strides = nullptr;
  const gert::ContinuousVector *pads = nullptr;
  const gert::ContinuousVector *dilations = nullptr;
  const int64_t *groups = nullptr;
  size_t idx = 0;
  strides = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  if (depthwise) {
    // DepthwiseConv2DBackpropFilter attr: strides, dilations, pads, data_format
    dilations = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    pads = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  } else {
    // Conv2DBackpropFilter attr: strides, pads, dilations, groups, data_format
    pads = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    dilations = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    groups = attrs->GetAttrPointer<int64_t>(idx++);
  }

  OP_LOGE_IF(strides == nullptr, false, op_name, "get strides from context fail.");
  OP_LOGE_IF(strides->GetSize() != kStrideDim, false, op_name, "strides of context dim len is invalid.");
  if (!depthwise) {
    OP_LOGE_IF(groups == nullptr, false, op_name, "get groups from context fail.");
    params.groups = static_cast<int32_t>(*groups);
  }

  OP_LOGE_IF(dilations == nullptr, false, op_name, "get dilations from context fail.");
  OP_LOGE_IF(dilations->GetSize() != kDilationDim, false, op_name, "dilations of context dim len is invalid.");
  OP_LOGE_IF(pads == nullptr, false, op_name, "get pads from context fail.");
  OP_LOGE_IF(pads->GetSize() != kPadsDim, false, op_name, "pads of context dim len is invalid.");

  auto y_desc = context->GetOutputDesc(0);
  OP_LOGE_IF(y_desc == nullptr, false, op_name, "get y_desc from context fail.");
  OP_LOGE_IF(y_desc->GetStorageFormat() != ge::FORMAT_NCHW, false, op_name,
             "current dw binary only support NCHW format.");

  const int64_t *strides_data = reinterpret_cast<const int64_t *>(strides->GetData());
  const int64_t *pads_data = reinterpret_cast<const int64_t *>(pads->GetData());
  const int64_t *dilations_data = reinterpret_cast<const int64_t *>(dilations->GetData());

  OP_LOGE_IF(strides_data[kNchwDimN] != 1 || strides_data[kNchwDimC] != 1, false, op_name,
             "strides N/C dim must be 1.");
  OP_LOGE_IF(dilations_data[kNchwDimN] != 1 || dilations_data[kNchwDimC] != 1, false, op_name,
             "dilations N/C dim must be 1.");

  params.pad_u = pads_data[kPadsUpDim];
  params.pad_d = pads_data[kPadsDownDim];
  params.pad_l = pads_data[kPadsLeftDim];
  params.pad_r = pads_data[kPadsRightDim];
  params.stride_h = strides_data[kNchwDimH];
  params.stride_w = strides_data[kNchwDimW];
  params.dilation_h = dilations_data[kNchwDimH];
  params.dilation_w = dilations_data[kNchwDimW];
  return true;
}

static bool TransformShape(const char *op_name, const gert::Shape &ori_shape, ge::Format ori_format, int64_t *shape) {
  // transform to nchw format
  OP_LOGE_IF(ori_shape.GetDimNum() != kOriShapeDim, false, op_name, "ori shape dim nums is invalid.");
  size_t idx = 0;
  if (ori_format == ge::FORMAT_NCHW) {
    shape[idx++] = ori_shape.GetDim(kPos0);
    shape[idx++] = ori_shape.GetDim(kPos1);
    shape[idx++] = ori_shape.GetDim(kPos2);
    shape[idx++] = ori_shape.GetDim(kPos3);
  } else if (ori_format == ge::FORMAT_HWCN) {
    shape[idx++] = ori_shape.GetDim(kPos3);
    shape[idx++] = ori_shape.GetDim(kPos2);
    shape[idx++] = ori_shape.GetDim(kPos0);
    shape[idx++] = ori_shape.GetDim(kPos1);
  } else if (ori_format == ge::FORMAT_NHWC) {
    shape[idx++] = ori_shape.GetDim(kPos0);
    shape[idx++] = ori_shape.GetDim(kPos3);
    shape[idx++] = ori_shape.GetDim(kPos1);
    shape[idx++] = ori_shape.GetDim(kPos2);
  } else {
    return false;
  }
  return true;
}

static bool ModifyPad(gert::TilingContext *context, bool depthwise, conv2d_dw::Conv2dBpFilterParas &params) {
  const auto attrs = context->GetAttrs();
  size_t padding_attr_idx = depthwise ? 4 : 5;  // DepthwiseConv2DBackpropFilter: 4, Conv2DBackpropFilter: 5
  if (attrs->GetAttrNum() <= padding_attr_idx) {
    OP_LOGD(context->GetNodeName(), "no padding attr, skip calc and check");
    return true;
  }

  int32_t filter_h = (params.kh - 1) * params.dilation_h + 1;
  int32_t filter_w = (params.kw - 1) * params.dilation_w + 1;
  const auto padding = attrs->GetAttrPointer<char>(padding_attr_idx);
  if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
    int32_t pad_h = max(Align(params.hi, params.stride_h) - params.stride_h + filter_h - params.hi, 0);
    int32_t pad_up = (pad_h >> 1L);
    int32_t pad_down = pad_h - pad_up;
    int32_t pad_w = max(Align(params.wi, params.stride_w) - params.stride_w + filter_w - params.wi, 0);
    int32_t pad_left = (pad_w >> 1L);
    int32_t pad_right = pad_w - pad_left;
    params.pad_u = pad_up;
    params.pad_d = pad_down;
    params.pad_l = pad_left;
    params.pad_r = pad_right;
    OP_LOGD(context->GetNodeName(), "set pads[%d, %d, %d, %d]", pad_up, pad_down, pad_left, pad_right);
  } else {
    int32_t ho_expect = (params.hi + params.pad_u + params.pad_d - filter_h) / params.stride_h + 1;
    int32_t wo_expect = (params.wi + params.pad_l + params.pad_r - filter_w) / params.stride_w + 1;
    OP_TILING_CHECK(ho_expect != params.ho || wo_expect != params.wo,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                          "check pads attrs failed, ho: %d, wo: %d, ho_expect: %d, wo_expect: %d",
                                          params.ho, params.wo, ho_expect, wo_expect),
                    return false);
  }
  return true;
}

static bool SetShapeParams(gert::TilingContext *context, size_t dedy_index, bool depthwise,
                           conv2d_dw::Conv2dBpFilterParas &params) {
  const auto op_name = context->GetNodeName();
  const auto fmap_desc = context->GetInputDesc(0);
  const auto dedy_desc = context->GetInputDesc(dedy_index);
  const auto filter_desc = context->GetOutputDesc(0);
  const auto fmap_shape = context->GetInputShape(0);
  const auto dedy_shape = context->GetInputShape(dedy_index);
  const auto filter_shape = context->GetOutputShape(0);

  int64_t dedy_shape_nchw[kOriShapeDim];
  int64_t filter_shape_nchw[kOriShapeDim];
  int64_t fmap_shape_nchw[kOriShapeDim];
  OP_LOGE_IF(!TransformShape(op_name, dedy_shape->GetOriginShape(), dedy_desc->GetOriginFormat(), dedy_shape_nchw),
             false, op_name, "Transform dedy shape fail.");
  OP_LOGE_IF(!TransformShape(op_name, fmap_shape->GetOriginShape(), fmap_desc->GetOriginFormat(), fmap_shape_nchw),
             false, op_name, "Transform fmap shape fail.");
  OP_LOGE_IF(
      !TransformShape(op_name, filter_shape->GetOriginShape(), filter_desc->GetOriginFormat(), filter_shape_nchw),
      false, op_name, "Transform filter shape fail.");

  params.batch = static_cast<int32_t>(dedy_shape_nchw[kNchwDimN]);
  params.ho = static_cast<int32_t>(dedy_shape_nchw[kNchwDimH]);
  params.wo = static_cast<int32_t>(dedy_shape_nchw[kNchwDimW]);
  params.co = static_cast<int32_t>(dedy_shape_nchw[kNchwDimC]);
  params.co0 = static_cast<int32_t>(kDefaultC0);
  params.co1 = static_cast<int32_t>((params.co + params.co0 - 1) / params.co0);

  params.hi = static_cast<int32_t>(fmap_shape_nchw[kNchwDimH]);
  params.wi = static_cast<int32_t>(fmap_shape_nchw[kNchwDimW]);
  params.ci = static_cast<int32_t>(fmap_shape_nchw[kNchwDimC]);
  params.ci0 = static_cast<int32_t>(kDefaultC0);
  params.ci1 = static_cast<int32_t>((params.ci + params.ci0 - 1) / params.ci0);

  params.kh = static_cast<int32_t>(filter_shape_nchw[kNchwDimH]);
  params.kw = static_cast<int32_t>(filter_shape_nchw[kNchwDimW]);
  if (filter_shape_nchw[kNchwDimC] == 0 || static_cast<int64_t>(params.ci) % filter_shape_nchw[kNchwDimC] != 0) {
    OP_LOGE(context->GetNodeName(), "fmap_channel(%d) %% filter_channel(%ld) != 0", params.ci,
            filter_shape_nchw[kNchwDimC]);
    return false;
  }

  int64_t groups = static_cast<int64_t>(params.ci) / filter_shape_nchw[kNchwDimC];
  if (params.groups == 1) {
    params.groups = static_cast<int32_t>(groups);
    OP_LOGD(context->GetNodeName(), "set groups=%d, fmap_channel(%d) / filter_channel(%ld)",
            params.groups, params.ci, filter_shape_nchw[kNchwDimC]);
  } else if (groups != static_cast<int64_t>(params.groups)) {
    OP_LOGE(context->GetNodeName(), "fmap_channel(%d) / filter_channel(%ld) != groups(%d)", params.ci,
            filter_shape_nchw[kNchwDimC], params.groups);
    return false;
  }

  return ModifyPad(context, depthwise, params);
}

static bool ParseOpInfo(gert::TilingContext *context, const Conv2DBackPropCompileInfo &compile_info, size_t dedy_index,
                        bool depthwise, conv2d_dw::Conv2dBpFilterParas &params) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!SetCacheTilingParamsFromOpDesc(context, params, depthwise), false, op_name,
             "Set cache tiling params failed.");
  OP_LOGE_IF(!SetShapeParams(context, dedy_index, depthwise, params), false, op_name, "Set shape params failed.");
  // dw tiling can use double times core num
  params.max_core_num = compile_info.core_num * kBlockDimRate;
  // check shape and attrs by params
  return true;
}

static bool UpdateRunInfoCube(const conv2d_dw::Conv2dDwTiling &tiling, const conv2d_dw::Conv2dBpFilterParas &params,
                              gert::TilingContext *context) {
  uint32_t block_dim = tiling.batch_dim * tiling.n_dim * tiling.h_dim * tiling.m_dim * tiling.group_dim;
  context->SetBlockDim(block_dim);
  context->SetTilingKey(stoull(tiling.tiling_id));  // always success, no need to capture exception
  auto tiling_data = context->GetRawTilingData();
  // transdata vars
  tiling_data->Append(params.batch);
  tiling_data->Append(params.ci);
  tiling_data->Append(params.hi);
  tiling_data->Append(params.wi);
  tiling_data->Append(params.co);
  tiling_data->Append(params.ho);
  tiling_data->Append(params.wo);
  tiling_data->Append(params.kh);
  tiling_data->Append(params.kw);
  // dw vars
  tiling_data->Append(params.ci1);
  tiling_data->Append(params.co1);
  tiling_data->Append(params.stride_h);
  tiling_data->Append(params.stride_w);
  tiling_data->Append(params.pad_u);
  tiling_data->Append(params.pad_d);
  tiling_data->Append(params.pad_l);
  tiling_data->Append(params.pad_r);
  tiling_data->Append(params.dilation_h);
  tiling_data->Append(params.dilation_w);
  tiling_data->Append(params.groups);
  // set tiling
  OP_LOGE_IF(params.ci0 == 0, false, params.op_type, "There is a div zero error. params:ci0 is zero.");
  OP_LOGE_IF(tiling.h_dim == 0, false, params.op_type, "There is a div zero error. tiling:h_dim is zero.");
  int32_t total_n = params.ci1 * params.kh * params.kw;
  int32_t single_core_k = ((params.ho * params.wo + params.ci0 - 1) / params.ci0) / tiling.h_dim;
  tiling_data->Append(tiling.group_dim);
  tiling_data->Append(tiling.batch_dim);
  // k_dim
  tiling_data->Append(tiling.h_dim);
  // batch_single_core
  OP_LOGE_IF(tiling.batch_dim == 0, false, params.op_type, "There is a div zero error. tiling:batch_dim is zero.");
  tiling_data->Append(params.batch / tiling.batch_dim);
  // n_single_core
  OP_LOGE_IF(tiling.n_dim * tiling.n_bl1 * tiling.n_l0 == 0, false, params.op_type,
             "There is a div zero error. tiling:n_dim * n_bl1 * n_l0 is zero.");
  tiling_data->Append(total_n / (tiling.n_dim * tiling.n_bl1 * tiling.n_l0));
  tiling_data->Append(tiling.n_dim);
  tiling_data->Append(tiling.n_bl1);
  // n_ub_l0_time = l0c_n / cub_n
  OP_LOGE_IF(tiling.n_cub == 0, false, params.op_type, "There is a div zero error. tiling:n_cub is zero.");
  tiling_data->Append(tiling.n_l0 / tiling.n_cub);
  // cub_n1
  tiling_data->Append(tiling.n_cub);
  tiling_data->Append(tiling.m_dim);
  // m_single_core
  OP_LOGE_IF(tiling.m_dim * tiling.m_al1 * tiling.m_l0 == 0, false, params.op_type,
             "There is a div zero error. tiling:tiling.m_dim * tiling.m_al1 * tiling.m_l0 is zero.");
  tiling_data->Append(params.co1 / tiling.m_dim / tiling.m_al1 / tiling.m_l0);
  tiling_data->Append(tiling.m_al1);
  tiling_data->Append(tiling.m_l0);
  tiling_data->Append(tiling.k_l0);
  // k_al1_factor
  OP_LOGE_IF(tiling.kal1_16 == 0, false, params.op_type, "There is a div zero error. tiling:kal1_16 is zero.");
  tiling_data->Append(single_core_k / tiling.kal1_16);
  // k_bl1_factor
  OP_LOGE_IF(tiling.kbl1_16 == 0, false, params.op_type, "There is a div zero error. tiling:kbl1_16 is zero.");
  tiling_data->Append(single_core_k / tiling.kbl1_16);
  // k_al0_factor
  OP_LOGE_IF(tiling.k_l0 == 0, false, params.op_type, "There is a div zero error. tiling:k_l0 is zero.");
  tiling_data->Append(tiling.kal1_16 / tiling.k_l0);
  // k_bl0_factor
  tiling_data->Append(tiling.kbl1_16 / tiling.k_l0);
  tiling_data->Append(tiling.kal1_16);
  tiling_data->Append(tiling.kbl1_16);
  // kl1_times
  tiling_data->Append(std::max(tiling.kal1_16, tiling.kbl1_16) / std::min(tiling.kal1_16, tiling.kbl1_16));
  tiling_data->Append(tiling.bl1_bound);
  tiling_data->Append(tiling.m_aub);
  tiling_data->Append(tiling.n_bub);
  tiling_data->Append(tiling.k_aub);
  tiling_data->Append(tiling.k_bub);
  tiling_data->Append(tiling.ho_bl1);
  // multi_n_ub_l1
  OP_LOGE_IF(tiling.n_bub == 0, false, params.op_type, "There is a div zero error. tiling:n_bub is zero.");
  tiling_data->Append(tiling.n_bl1 * tiling.n_l0 / tiling.n_bub);
  // multi_m_ub_l1
  OP_LOGE_IF(tiling.m_aub == 0, false, params.op_type, "There is a div zero error. tiling:m_aub is zero.");
  tiling_data->Append(tiling.m_al1 * tiling.m_l0 / tiling.m_aub);
  // multi_k_aub_l1
  OP_LOGE_IF(tiling.k_aub == 0, false, params.op_type, "There is a div zero error. tiling:k_aub is zero.");
  tiling_data->Append(tiling.kal1_16 / tiling.k_aub);
  // multi_k_bub_l1
  OP_LOGE_IF(tiling.k_bub == 0, false, params.op_type, "There is a div zero error. tiling:k_bub is zero.");
  tiling_data->Append(tiling.kbl1_16 / tiling.k_bub);
  return true;
}

static bool BinaryTiling(gert::TilingContext *context, const Conv2DBackPropCompileInfo &compile_info, size_t dedy_index,
                         bool depthwise) {
  conv2d_dw::Conv2dBpFilterParas params;
  params.op_type = context->GetNodeType();
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!ParseOpInfo(context, compile_info, dedy_index, depthwise, params), false, op_name,
             "Parse cache tiling params failed.");
  OP_LOGE_IF(!CheckCacheTilingParams(params), false, op_name, "Check cache tiling params failed.");
  conv2d_dw::Conv2dDwCacheTiling cacheTiling(params);
  conv2d_dw::Conv2dDwTiling tiling;
  OP_LOGE_IF(!cacheTiling.GenTiling(tiling), false, op_name, "GenTiling failed!");
  return UpdateRunInfoCube(tiling, params, context);
}

static size_t InitVarsValues(uint32_t var_bit_flags, const gert::Shape &fmap_shape, const gert::Shape &dedy_shape,
                             gert::Shape &var_value, int64_t *shape_for_range_match) {
  if ((var_bit_flags & kVarBatch) != 0) {
    var_value.AppendDim(fmap_shape.GetDim(kConv2dNDim));
  }

  if ((var_bit_flags & kVarFmapH) != 0) {
    var_value.AppendDim(fmap_shape.GetDim(kConv2dHDim));
    var_value.AppendDim(dedy_shape.GetDim(kConv2dHDim));
  }

  if ((var_bit_flags & kVarFmapW) != 0) {
    var_value.AppendDim(fmap_shape.GetDim(kConv2dWDim));
    var_value.AppendDim(dedy_shape.GetDim(kConv2dWDim));
  }

  size_t idx = 0;
  shape_for_range_match[idx++] = fmap_shape.GetDim(kConv2dNDim);
  if (var_value.GetDimNum() > 1) {  // not only dynamic batch
    shape_for_range_match[idx++] = fmap_shape.GetDim(kConv2dHDim);
    shape_for_range_match[idx++] = fmap_shape.GetDim(kConv2dWDim);
  }

  return idx;
}

ge::graphStatus TilingForConv2DDw(gert::TilingContext *context, bool depthwise) {
  const auto compile_info = reinterpret_cast<const Conv2DBackPropCompileInfo *>(context->GetCompileInfo());
  OP_TILING_CHECK(compile_info == nullptr, CUBE_INNER_ERR_REPORT("Conv2DBackpropFilter", "compile_info is null"),
                  return ge::GRAPH_FAILED);

  size_t dedy_index = 2;  // in UB Fusion, it is 1, can't detect it now, wait for GE ready
  const auto fmap_desc = context->GetInputDesc(0);
  const auto dedy_desc = context->GetInputDesc(dedy_index);
  const auto filter_desc = context->GetOutputDesc(0);
  const auto fmap_shape = context->GetInputShape(0);
  const auto dedy_shape = context->GetInputShape(dedy_index);
  const auto filter_shape = context->GetOutputShape(0);

  OP_TILING_CHECK(fmap_desc == nullptr || dedy_desc == nullptr || filter_desc == nullptr || fmap_shape == nullptr ||
                      dedy_shape == nullptr || filter_shape == nullptr,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get fmap/out_backprop/filter shape/tensor"),
                  return ge::GRAPH_FAILED);

  if (compile_info->repo_binary_flag) {
    OP_TILING_CHECK(!BinaryTiling(context, *compile_info, dedy_index, depthwise),
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "binary tiling process fail"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }

  const auto &fmap_storage_shape = fmap_shape->GetStorageShape();
  const auto &dedy_storage_shape = dedy_shape->GetStorageShape();

  bool unvalid_size = context->GetComputeNodeInfo()->GetInputsNum() < kConv2dDwInputSizeLimit ||
                      context->GetComputeNodeInfo()->GetOutputsNum() == 0 ||
                      fmap_storage_shape.GetDimNum() < kConv2dDimNumLimit ||
                      dedy_storage_shape.GetDimNum() < kConv2dDimNumLimit;

  OP_TILING_CHECK(unvalid_size, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "the size is unvalid."),
                  return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "Current format is %s, Ori format is %s",
          ge::TypeUtils::FormatToSerialString(fmap_desc->GetStorageFormat()).c_str(),
          ge::TypeUtils::FormatToSerialString(fmap_desc->GetOriginFormat()).c_str());

  gert::Shape var_value;
  int64_t shape_for_range_match[3];  // 3: nhw
  size_t dim_num = InitVarsValues(compile_info->var_bit_flags, fmap_storage_shape, dedy_storage_shape, var_value,
                                  shape_for_range_match);
  return CubeTiling(shape_for_range_match, dim_num, var_value, *compile_info, context);
}

ge::graphStatus TilingForConv2DBpFilter(gert::TilingContext *context) {
  return TilingForConv2DDw(context, false);
}

ge::graphStatus TilingForDepthwiseConv2DBackpropFilter(gert::TilingContext *context) {
  return TilingForConv2DDw(context, true);
}

IMPL_OP(Conv2DBackpropFilter)
    .Tiling(TilingForConv2DBpFilter)
    .TilingParse<Conv2DBackPropCompileInfo>(ParseConv2DBackpropCompileInfo);

IMPL_OP(DepthwiseConv2DBackpropFilter)
    .Tiling(TilingForDepthwiseConv2DBackpropFilter)
    .TilingParse<Conv2DBackPropCompileInfo>(ParseConv2DBackpropCompileInfo);
}  // namespace optiling
