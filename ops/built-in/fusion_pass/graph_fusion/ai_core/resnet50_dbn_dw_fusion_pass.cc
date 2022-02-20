/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file resnet50_dbn_dw_fusion_pass.cc
 * \brief resnet50_dbn_dw_fusion_pass pass
 */

#include "resnet50_dbn_dw_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "fp16_t.hpp"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "common/util/platform_info.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_INPUTS0 = "input0";
static const string PATTERN_DBN = "BNTrainingReduceGrad";
static const string PATTERN_CONV2DBPFILTER = "Conv2DBackpropFilterD";
static const string PATTERN_FUSEDDBNDW = "FusedDbnDw";
static const uint32_t kSupportAicoreNum = 32;
static const vector<int64_t> kSupportBatch = {32, 256};
static const vector<int64_t> Batch256AddCase = {
  256, 1024, 14, 14, 14, 14, 1, 1
};
static const vector<vector<int64_t>> kSupportCases = {
    // c_in, c_out, x_h, x_w, y_h, y_w, k_h, k_w
    {64, 256, 56, 56, 56, 56, 1, 1},
    {256, 64, 56, 56, 56, 56, 1, 1},
    {3, 64, 224, 224, 112, 112, 7, 7},
    {512, 128, 28, 28, 28, 28, 1, 1},
    {64, 64, 56, 56, 56, 56, 3, 3},
    {256, 512, 56, 56, 28, 28, 1, 1},
    {128, 512, 28, 28, 28, 28, 1, 1},
    {256, 128, 56, 56, 56, 56, 1, 1},
    {64, 64, 56, 56, 56, 56, 1, 1},
};

vector<FusionPattern*> Resnet50DbnDwFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (nothrow) FusionPattern("Resnet50DbnDwFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "new pattern obj failed"),
                    return patterns);
  pattern->AddOpDesc(PATTERN_DBN, {PATTERN_DBN})
          .AddOpDesc(PATTERN_CONV2DBPFILTER, {PATTERN_CONV2DBPFILTER})
          .AddOpDesc(PATTERN_INPUTS0)
          .SetInputs(PATTERN_CONV2DBPFILTER, {PATTERN_INPUTS0, PATTERN_DBN})
          .SetOutput(PATTERN_CONV2DBPFILTER);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern Resnet50DbnDwFusionPass success.");
  return patterns;
}

Status Resnet50DbnDwFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping,
                                       vector<NodePtr>& fusion_nodes) {
  OP_LOGD("Start to fuse DBN and DW.");
  // check the platform
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                    platform_info, opti_compilation_info) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get platform_info failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(platform_info.soc_info.ai_core_cnt != kSupportAicoreNum,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "this platform not support dw&dbn fusion."),
                    return NOT_CHANGED);

  // get the origin Node from mapping
  NodePtr dbn_node = GetNodeFromMapping(PATTERN_DBN, mapping);
  NodePtr dw_node = GetNodeFromMapping(PATTERN_CONV2DBPFILTER, mapping);
  FUSION_PASS_CHECK(dbn_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Dbn node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(dw_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Dw node is null, fusion failed."),
                    return PARAM_INVALID);
  OpDescPtr dw_op_desc = dw_node->GetOpDesc();
  OpDescPtr dbn_op_desc = dbn_node->GetOpDesc();

  // create a new node which type is PATTERN_FUSEDDBNDW
  OpDescPtr fused_dbn_dw_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    fused_dbn_dw_desc = make_shared<ge::OpDesc>(dbn_node->GetName() + "_Conv2DBackpropFilterD_FUSED_layer",
                                                PATTERN_FUSEDDBNDW),
    return FAILED);

  // set the new Node OpDesc by origin Node
  SetOpDesc(dw_op_desc, dbn_op_desc, fused_dbn_dw_desc);
  FUSION_PASS_CHECK(CheckSupportCase(dw_op_desc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "this case not support dw&dbn fusion."),
                    return NOT_CHANGED);

  // add the node and push back it to fusion_nodes
  NodePtr fused_dbn_dw_node = graph.AddNode(fused_dbn_dw_desc);
  FUSION_PASS_CHECK(fused_dbn_dw_node == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedDbnDw Node is null, fusion failed."),
                    return FAILED);
  fusion_nodes.push_back(fused_dbn_dw_node);

  // handle the edges, connect all inputs edges and all outputs edges to the new node
  FUSION_PASS_CHECK(GraphUtils::AddEdge(dw_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                        fused_dbn_dw_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "add edge from dw's input to fusednode's input failed"),
                    return FAILED);
  for (size_t i = 0; i < dbn_node->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(GraphUtils::AddEdge(dbn_node->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                          fused_dbn_dw_node->GetInDataAnchor(i+1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "add edge from dbn's input to fusednode's input failed"),
                      return FAILED);
  }
  OutDataAnchorPtr dbn_out_anchor_ptr = dbn_node->GetOutDataAnchor(0);
  OutDataAnchorPtr dw_out_anchor_ptr = dw_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(dbn_out_anchor_ptr == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Dbn out anchor is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(dw_out_anchor_ptr == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Dw out anchor is null, fusion failed."),
                    return PARAM_INVALID);
  for (auto post_anchor_ptr0 : dbn_out_anchor_ptr->GetPeerInDataAnchors()) {
    post_anchor_ptr0->UnlinkAll();
    FUSION_PASS_CHECK(GraphUtils::AddEdge(fused_dbn_dw_node->GetOutDataAnchor(0), post_anchor_ptr0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "add edge between fused node and dbn's next node failed"),
                      return FAILED);
  }
  for (auto post_anchor_ptr0 : dw_out_anchor_ptr->GetPeerInDataAnchors()) {
    post_anchor_ptr0->UnlinkAll();
    FUSION_PASS_CHECK(GraphUtils::AddEdge(fused_dbn_dw_node->GetOutDataAnchor(1), post_anchor_ptr0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "add edge between fused node and dw's next node failed"),
                      return FAILED);
  }

  // delete dbn_node and dw_node
  FUSION_PASS_CHECK(
    graph.RemoveNode(dbn_node) == GRAPH_FAILED,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Removing Dbn Node is failed."),
    return FAILED);

  FUSION_PASS_CHECK(
    graph.RemoveNode(dw_node) == GRAPH_FAILED,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Removing Dw Node is failed."),
    return FAILED);

  return SUCCESS;
}

Status Resnet50DbnDwFusionPass::SetOpDesc(OpDescPtr &dw_op_desc, OpDescPtr &dbn_op_desc,
                                          OpDescPtr &fused_dbn_dw_desc) {
  GeTensorDesc dw_input_desc = dw_op_desc->GetInputDesc(0);
  GeTensorDesc dw_output_desc = dw_op_desc->GetOutputDesc(0);
  GeTensorDesc dbn_output_desc = dbn_op_desc->GetOutputDesc(0);
  FUSION_PASS_CHECK(fused_dbn_dw_desc->AddInputDesc(dw_input_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dw input failed"),
                    return FAILED);
  for (auto dbn_input_desc : dbn_op_desc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(fused_dbn_dw_desc->AddInputDesc(dbn_input_desc) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add dbn input failed"),
                      return FAILED);
  }
  FUSION_PASS_CHECK(fused_dbn_dw_desc->AddOutputDesc(dbn_output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dbn output failed"),
                    return FAILED);
  FUSION_PASS_CHECK(fused_dbn_dw_desc->AddOutputDesc(dw_output_desc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add dw output failed"),
                    return FAILED);

  vector<int64_t> filter_size_index;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(dw_op_desc, "filter_size", filter_size_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get filter_size failed"),
                    return PARAM_INVALID);
  AttrUtils::SetListInt(fused_dbn_dw_desc, "filter_size", filter_size_index);

  vector<int64_t> strides_index;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(dw_op_desc, "strides", strides_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get strides failed"),
                    return PARAM_INVALID);
  AttrUtils::SetListInt(fused_dbn_dw_desc, "strides", strides_index);

  vector<int64_t> pads_index;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(dw_op_desc, "pads", pads_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get pads failed"),
                    return PARAM_INVALID);
  AttrUtils::SetListInt(fused_dbn_dw_desc, "pads", pads_index);

  vector<int64_t> dilations_index;
  FUSION_PASS_CHECK(!AttrUtils::GetListInt(dw_op_desc, "dilations", dilations_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get dilations failed"),
                    return PARAM_INVALID);
  AttrUtils::SetListInt(fused_dbn_dw_desc, "dilations", dilations_index);

  int64_t groups_index;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(dw_op_desc, "groups", groups_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get groups failed"),
                    return PARAM_INVALID);
  AttrUtils::SetInt(fused_dbn_dw_desc, "groups", groups_index);

  string data_format_index;
  FUSION_PASS_CHECK(!AttrUtils::GetStr(dw_op_desc, "data_format", data_format_index),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get data_format failed"),
                    return PARAM_INVALID);
  AttrUtils::SetStr(fused_dbn_dw_desc, "data_format", data_format_index);

  float epsilon_val;
  FUSION_PASS_CHECK(!AttrUtils::GetFloat(dbn_op_desc, "epsilon", epsilon_val),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get epsilon failed"),
                    return PARAM_INVALID);
  AttrUtils::SetFloat(fused_dbn_dw_desc, "epsilon", epsilon_val);

  return SUCCESS;
}

Status Resnet50DbnDwFusionPass::CheckSupportCase(OpDescPtr &dw_op_desc) {
  int64_t batch = 0;
  int64_t group = 0;
  int64_t c_in = 0;
  int64_t c_out = 0;
  int64_t fmap_h = 0;
  int64_t fmap_w = 0;
  int64_t dy_h = 0;
  int64_t dy_w = 0;
  int64_t filter_h = 0;
  int64_t filter_w = 0;
  vector<int64_t> params = {};

  GeTensorDesc dw_fmap_desc = dw_op_desc->GetInputDesc(0);
  vector<int64_t> fmap_dim_info = dw_fmap_desc.GetShape().GetDims();
  Format fmap_origin_format = dw_fmap_desc.GetOriginFormat();
  vector<int64_t> fmap_nchw = GetNchwVec(fmap_dim_info, fmap_origin_format);
  if (fmap_nchw.size() != 4) {
    return FAILED;
  } else {
    batch = fmap_nchw[0];
    c_in = fmap_nchw[1];
    fmap_h = fmap_nchw[2];
    fmap_w = fmap_nchw[3];
  }
  GeTensorDesc dw_dy_desc = dw_op_desc->GetInputDesc(1);
  vector<int64_t> dy_dim_info = dw_dy_desc.GetShape().GetDims();
  Format dy_origin_format = dw_dy_desc.GetOriginFormat();
  vector<int64_t> dy_nchw = GetNchwVec(dy_dim_info, dy_origin_format);
  if (dy_nchw.size() != 4) {
    return FAILED;
  } else {
    c_out = dy_nchw[1];
    dy_h = dy_nchw[2];
    dy_w = dy_nchw[3];
  }
  GeTensorDesc dw_output_desc = dw_op_desc->GetOutputDesc(0);
  vector<int64_t> output_dim_info = dw_output_desc.GetShape().GetDims();
  Format output_origin_format = dw_output_desc.GetOriginFormat();
  vector<int64_t> filter_nchw = GetNchwVec(output_dim_info, output_origin_format);
  if (filter_nchw.size() != 4) {
    return FAILED;
  } else {
    filter_h = filter_nchw[2];
    filter_w = filter_nchw[3];
  }
  AttrUtils::GetInt(dw_op_desc, "groups", group);

  // c_in, c_out, x_h, x_w, y_h, y_w, k_h, k_w
  params.push_back(c_in);
  params.push_back(c_out);
  params.push_back(fmap_h);
  params.push_back(fmap_w);
  params.push_back(dy_h);
  params.push_back(dy_w);
  params.push_back(filter_h);
  params.push_back(filter_w);

  int64_t batch_march = count(kSupportBatch.begin(), kSupportBatch.end(), batch);
  if (batch_march == 0 || group != 1) {
    return FAILED;
  }
  for (auto case_now : kSupportCases) {
    if (params == case_now) {
      return SUCCESS;
    }
  }
  if (batch == 256 && params == Batch256AddCase) {
    return SUCCESS;
  }
  return FAILED;
}

vector<int64_t> Resnet50DbnDwFusionPass::GetNchwVec(vector<int64_t> &dim_info,
                                                    Format &origin_format){
  int64_t param_n;
  int64_t param_c;
  int64_t param_h;
  int64_t param_w;
  vector<int64_t> params_nchw = {};
  if (dim_info.size() == 4) {
    if (origin_format == FORMAT_NHWC){
      param_n = dim_info[0];
      param_h = dim_info[1];
      param_w = dim_info[2];
      param_c = dim_info[3];
    } else if (origin_format == FORMAT_NCHW){
      param_n = dim_info[0];
      param_c = dim_info[1];
      param_h = dim_info[2];
      param_w = dim_info[3];
    } else if (origin_format == FORMAT_HWCN){
      param_h = dim_info[0];
      param_w = dim_info[1];
      param_c = dim_info[2];
      param_n = dim_info[3];
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "OriginFormat only support NHWC and NCHW and HWCN");
      return params_nchw;
    }
  } else {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_info size is not right");
    return params_nchw;
  }
  params_nchw.push_back(param_n);
  params_nchw.push_back(param_c);
  params_nchw.push_back(param_h);
  params_nchw.push_back(param_w);
  return params_nchw;
}

REGISTER_PASS("Resnet50DbnDwFusionPass", BUILT_IN_GRAPH_PASS, Resnet50DbnDwFusionPass);
} // namespace fe
