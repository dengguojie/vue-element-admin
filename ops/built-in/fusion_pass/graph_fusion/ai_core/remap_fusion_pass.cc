/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file remap_fusion_pass.cpp
 * \brief remap fusion pass
 */
#include "remap_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const float FLOAT_NUM_ZERO = 0;
static const string PATTERN_FUSEDNODE = "FusedNodeRemap";
static const string FUSED_NODE = "Remap";

vector<FusionPattern*> RemapFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("RemapFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status RemapFusionPass::CreateSplitNode(ge::NodePtr& split_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                        vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> split_desc = nullptr;
  std::string split_desc_name = fused_node->GetName() + "_splitD";
  split_desc = std::make_shared<ge::OpDesc>(split_desc_name, "SplitD");
  FUSION_PASS_CHECK(split_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "split_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  AttrUtils::SetInt(split_desc, "split_dim", 1);
  AttrUtils::SetInt(split_desc, "num_split", 2);
  // split input tensorDesc [N2HW]
  ge::GeTensorDesc input_desc = temp_desc;
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();
  input_shape[1] = 2;
  ge::GeShape new_shape(input_shape);
  input_desc.SetShape(new_shape);
  input_desc.SetOriginShape(new_shape);
  FUSION_PASS_CHECK(split_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for Split fail, fusion failed."), return NOT_CHANGED);

  // split output tensorDesc [N1HW]
  for (int64_t i = 0; i < 2; i++) {
    FUSION_PASS_CHECK(split_desc->AddOutputDesc("y" + std::to_string(i + 1), temp_desc) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for split fail, fusion failed."), return NOT_CHANGED);
  }

  split_node = graph.AddNode(split_desc);
  new_nodes.push_back(split_node);
  return SUCCESS;
}

Status RemapFusionPass::CreateFloorxNode(ge::NodePtr& floorx_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> floorx_desc = nullptr;
  std::string floorx_desc_name = fused_node->GetName() + "_floorx";
  floorx_desc = std::make_shared<ge::OpDesc>(floorx_desc_name, "Floor");
  FUSION_PASS_CHECK(floorx_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "floorx_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // floorx input tensorDesc
  FUSION_PASS_CHECK(floorx_desc->AddInputDesc("x", temp_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for floorx fail, fusion failed."), return NOT_CHANGED);

  // floorx output tensorDesc
  auto output_desc = temp_desc;
  FUSION_PASS_CHECK(floorx_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for floorx fail, fusion failed."), return NOT_CHANGED);

  floorx_node = graph.AddNode(floorx_desc);
  new_nodes.push_back(floorx_node);
  return SUCCESS;
}

Status RemapFusionPass::CreateCeilxNode(ge::NodePtr& ceilx_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                        vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> ceilx_desc = nullptr;
  std::string ceilx_desc_name = fused_node->GetName() + "_ceilx";
  ceilx_desc = std::make_shared<ge::OpDesc>(ceilx_desc_name, "Ceil");
  FUSION_PASS_CHECK(ceilx_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ceilx_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // floorx input tensorDesc
  FUSION_PASS_CHECK(ceilx_desc->AddInputDesc("x", temp_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for ceilx fail, fusion failed."), return NOT_CHANGED);

  // floorx output tensorDesc
  auto output_desc = temp_desc;
  FUSION_PASS_CHECK(ceilx_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for ceilx fail, fusion failed."), return NOT_CHANGED);

  ceilx_node = graph.AddNode(ceilx_desc);
  new_nodes.push_back(ceilx_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateFlooryNode(ge::NodePtr& floory_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> floory_desc = nullptr;
  std::string floory_desc_name = fused_node->GetName() + "_floory";
  floory_desc = std::make_shared<ge::OpDesc>(floory_desc_name, "Floor");
  FUSION_PASS_CHECK(floory_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "floory_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // floorx input tensorDesc
  FUSION_PASS_CHECK(floory_desc->AddInputDesc("x", temp_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for floory fail, fusion failed."), return NOT_CHANGED);

  // floorx output tensorDesc
  auto output_desc = temp_desc;
  FUSION_PASS_CHECK(floory_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for floory fail, fusion failed."), return NOT_CHANGED);

  floory_node = graph.AddNode(floory_desc);
  new_nodes.push_back(floory_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateCeilyNode(ge::NodePtr& ceily_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                        vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> ceily_desc = nullptr;
  std::string ceily_desc_name = fused_node->GetName() + "_ceily";
  ceily_desc = std::make_shared<ge::OpDesc>(ceily_desc_name, "Ceil");
  FUSION_PASS_CHECK(ceily_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ceily_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // ceily input tensorDesc
  FUSION_PASS_CHECK(ceily_desc->AddInputDesc("x", temp_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for ceily fail, fusion failed."), return NOT_CHANGED);

  // ceily output tensorDesc
  auto output_desc = temp_desc;
  FUSION_PASS_CHECK(ceily_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for ceily fail, fusion failed."), return NOT_CHANGED);

  ceily_node = graph.AddNode(ceily_desc);
  new_nodes.push_back(ceily_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateMulsx1Node(ge::NodePtr& mulsx1_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc,
                                         float& val) const {
  std::shared_ptr<ge::OpDesc> mulsx1_desc = nullptr;
  std::string mulsx1_desc_name = fused_node->GetName() + "_mulsx1";
  mulsx1_desc = std::make_shared<ge::OpDesc>(mulsx1_desc_name, "Muls");
  FUSION_PASS_CHECK(mulsx1_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "mulsx1_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // Mulsx1 input tensorDesc
  auto input_desc = temp_desc;
  input_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsx1_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for mulsx1 fail, fusion failed."), return NOT_CHANGED);

  // Mulsx1 output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsx1_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for mulsx1 fail, fusion failed."), return NOT_CHANGED);

  AttrUtils::SetFloat(mulsx1_desc, "value", val);
  mulsx1_node = graph.AddNode(mulsx1_desc);
  new_nodes.push_back(mulsx1_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateMulsx2Node(ge::NodePtr& mulsx2_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc,
                                         float& val) const {
  std::shared_ptr<ge::OpDesc> mulsx2_desc = nullptr;
  std::string mulsx2_desc_name = fused_node->GetName() + "_mulsx2";
  mulsx2_desc = std::make_shared<ge::OpDesc>(mulsx2_desc_name, "Muls");
  FUSION_PASS_CHECK(mulsx2_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "mulsx2_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // Mulsx2 input tensorDesc
  auto input_desc = temp_desc;
  input_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsx2_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for mulsx2 fail, fusion failed."), return NOT_CHANGED);

  // Mulsx2 output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsx2_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for mulsx2 fail, fusion failed."), return NOT_CHANGED);

  AttrUtils::SetFloat(mulsx2_desc, "value", val);
  mulsx2_node = graph.AddNode(mulsx2_desc);
  new_nodes.push_back(mulsx2_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateMulsy1Node(ge::NodePtr& mulsy1_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc,
                                         float& val) const {
  std::shared_ptr<ge::OpDesc> mulsy1_desc = nullptr;
  std::string mulsy1_desc_name = fused_node->GetName() + "_mulsy1";
  mulsy1_desc = std::make_shared<ge::OpDesc>(mulsy1_desc_name, "Muls");
  FUSION_PASS_CHECK(mulsy1_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "mulsy1_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // Mulsy1 input tensorDesc
  auto input_desc = temp_desc;
  input_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsy1_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for mulsy1 fail, fusion failed."), return NOT_CHANGED);

  // Mulsy1 output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsy1_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for mulsy1 fail, fusion failed."), return NOT_CHANGED);
  AttrUtils::SetFloat(mulsy1_desc, "value", val);
  mulsy1_node = graph.AddNode(mulsy1_desc);
  new_nodes.push_back(mulsy1_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateMulsy2Node(ge::NodePtr& mulsy2_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc,
                                         float& val) const {
  std::shared_ptr<ge::OpDesc> mulsy2_desc = nullptr;
  std::string mulsy2_desc_name = fused_node->GetName() + "_mulsy2";
  mulsy2_desc = std::make_shared<ge::OpDesc>(mulsy2_desc_name, "Muls");
  FUSION_PASS_CHECK(mulsy2_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "mulsy1_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // Mulsy2 input tensorDesc
  auto input_desc = temp_desc;
  input_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsy2_desc->AddInputDesc("x", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for mulsy1 fail, fusion failed."), return NOT_CHANGED);

  // Mulsy2 output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(mulsy2_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for mulsy1 fail, fusion failed."), return NOT_CHANGED);
  AttrUtils::SetFloat(mulsy2_desc, "value", val);
  mulsy2_node = graph.AddNode(mulsy2_desc);
  new_nodes.push_back(mulsy2_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateAddNode(const std::string name, ge::NodePtr& add_node, ge::NodePtr& fused_node,
                                      ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                      ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> add_desc = nullptr;
  std::string add_desc_name = fused_node->GetName() + name;
  add_desc = std::make_shared<ge::OpDesc>(add_desc_name, "Add");
  FUSION_PASS_CHECK(add_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // add input tensorDesc
  auto input_desc = temp_desc;
  input_desc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(add_desc->AddInputDesc("x1", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for %s fail, fusion failed.", add_desc_name.c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(add_desc->AddInputDesc("x2", input_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input y for %s fail, fusion failed.", add_desc_name.c_str()),
                    return NOT_CHANGED);

  // add output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(add_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput z for %s after fail, fusion failed.",
                            add_desc_name.c_str()),
                    return NOT_CHANGED);

  add_node = graph.AddNode(add_desc);
  new_nodes.push_back(add_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateCastNode(const std::string name, ge::NodePtr& cast_node, ge::NodePtr& fused_node,
                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                       ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> cast_desc = nullptr;
  std::string cast_desc_name = fused_node->GetName() + name;
  cast_desc = std::make_shared<ge::OpDesc>(cast_desc_name, "Cast");
  FUSION_PASS_CHECK(cast_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "cast_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // cast input tensorDesc
  FUSION_PASS_CHECK(cast_desc->AddInputDesc("x", temp_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for %s fail, fusion failed.", cast_desc_name.c_str()),
                    return NOT_CHANGED);

  // cast output tensorDesc
  auto output_desc = temp_desc;
  output_desc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(cast_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput z for %s after fail, fusion failed.",
                            cast_desc_name.c_str()),
                    return NOT_CHANGED);

  AttrUtils::SetInt(cast_desc, "dst_type", DT_INT32);
  cast_node = graph.AddNode(cast_desc);
  new_nodes.push_back(cast_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateConcatNode(ge::NodePtr& concat_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const {
  std::shared_ptr<ge::OpDesc> concat_desc = nullptr;
  std::string concat_desc_name = fused_node->GetName() + "_concat";
  concat_desc = std::make_shared<ge::OpDesc>(concat_desc_name, "ConcatD");
  FUSION_PASS_CHECK(concat_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "concat_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  auto input_desc = temp_desc;
  input_desc.SetDataType(ge::DT_INT32);
  // Concat input tensorDesc
  for (int64_t i = 0; i < 4; i++) {
    FUSION_PASS_CHECK(concat_desc->AddInputDesc("input_values" + std::to_string(i + 1), input_desc) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "add input for concat fail, fusion failed."), return NOT_CHANGED);
  }
  // Concat output tensorDesc
  auto output_desc = temp_desc;
  vector<int64_t> tmp_shape = output_desc.GetShape().GetDims();
  std::vector<int64_t> output_shape = {tmp_shape[0], 4, tmp_shape[2], tmp_shape[3]};
  ge::GeShape new_shape(output_shape);
  output_desc.SetShape(new_shape);
  output_desc.SetOriginShape(new_shape);
  output_desc.SetDataType(ge::DT_INT32);
  FUSION_PASS_CHECK(concat_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput y for concat fail, fusion failed."), return NOT_CHANGED);
  // attr
  ge::AttrUtils::SetInt(concat_desc, "concat_dim", 1);
  ge::AttrUtils::SetInt(concat_desc, "N", 4);
  concat_node = graph.AddNode(concat_desc);
  new_nodes.push_back(concat_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateRemapOffsetsNode(ge::NodePtr& offsets_node, ge::NodePtr& fused_node,
                                               ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                               ge::GeTensorDesc& input0_desc, ge::GeTensorDesc& input1_desc) const {
  std::shared_ptr<ge::OpDesc> offsets_desc = nullptr;
  std::string offsets_desc_name = fused_node->GetName() + "_RemapOffsets";
  offsets_desc = std::make_shared<ge::OpDesc>(offsets_desc_name, "IMGWarpOffsets");
  FUSION_PASS_CHECK(offsets_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "RemapOffsets_desc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(offsets_desc->AddInputDesc("images", input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input offsets for offsets_node fail, fusion failed."),
                    return NOT_CHANGED);
  auto in1_desc = input1_desc;
  vector<int64_t> shape = in1_desc.GetShape().GetDims();
  std::vector<int64_t> input1_shape = {shape[0], 4, shape[1], shape[2]};
  ge::GeShape tmp_shape(input1_shape);
  in1_desc.SetShape(tmp_shape);
  in1_desc.SetOriginShape(tmp_shape);
  in1_desc.SetDataType(DT_INT32);
  FUSION_PASS_CHECK(offsets_desc->AddInputDesc("offsets", in1_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "add input offsets for offsets_node offsets_node fail, fusion failed."),
                    return NOT_CHANGED);

  std::vector<int64_t> input0_shape = input0_desc.GetShape().GetDims();
  // the output shape is [N,4,H,W,C]
  ge::GeTensorDesc output_desc = input0_desc;
  vector<int64_t> output_shape = input1_shape;
  output_shape.push_back(input0_shape[3]);
  ge::GeShape new_shape(output_shape);
  output_desc.SetShape(new_shape);
  output_desc.SetOriginShape(new_shape);
  FUSION_PASS_CHECK(offsets_desc->AddOutputDesc("warp_images", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add output warp_images for offsets_node fail, fusion failed."),
                    return NOT_CHANGED);
  offsets_node = graph.AddNode(offsets_desc);
  new_nodes.push_back(offsets_node);

  return SUCCESS;
}

Status RemapFusionPass::CreateRemapResizeNode(ge::NodePtr& resize_node, ge::NodePtr& fused_node,
                                              ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                              ge::GeTensorDesc& input0_desc, ge::GeTensorDesc& input1_desc,
                                              ge::GeTensorDesc& output_desc) const {
  std::shared_ptr<ge::OpDesc> resize_desc = nullptr;
  std::string resize_desc_name = fused_node->GetName() + "_RemapResize";
  resize_desc = std::make_shared<ge::OpDesc>(resize_desc_name, "IMGWarpResize");
  FUSION_PASS_CHECK(resize_desc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "RemapOffsets_desc is null, fusion failed."),
                    return NOT_CHANGED);
  ge::GeTensorDesc resize_input0_desc = input0_desc;
  vector<int64_t> input0_shape = resize_input0_desc.GetShape().GetDims();
  input0_shape.insert(input0_shape.begin() + 1, 4);
  auto input1_shape = input1_desc.GetShape().GetDims();
  input0_shape[2] = input1_shape[1];
  input0_shape[3] = input1_shape[2];
  ge::GeShape new_shape(input0_shape);
  resize_input0_desc.SetShape(new_shape);
  resize_input0_desc.SetOriginShape(new_shape);
  auto output_shape = output_desc.GetShape().GetDims();
  auto output_shape_new = output_shape;
  output_shape_new[1] = output_shape[3];
  output_shape_new[2] = output_shape[1];
  output_shape_new[3] = output_shape[2];
  output_desc.SetShape(ge::GeShape(output_shape_new));
  output_desc.SetOriginShape(ge::GeShape(output_shape_new));
  FUSION_PASS_CHECK(resize_desc->AddInputDesc("img", resize_input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input img for resize_node fail, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      resize_desc->AddInputDesc("map_offset", input1_desc) != SUCCESS,
      OP_LOGW(FUSED_OP_TYPE.c_str(), "add input map_offset for resize_node offsets_node fail, fusion failed."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(resize_desc->AddOutputDesc("map_img", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add output map_img for resize_node fail, fusion failed."),
                    return NOT_CHANGED);
  resize_node = graph.AddNode(resize_desc);
  new_nodes.push_back(resize_node);

  return SUCCESS;
}

Status RemapFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define RemapFusionPass fusion begin.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "get fused_node failed, fusion failed."),
                    return NOT_CHANGED);

  // get input 1 shape
  ge::GeTensorDesc img_input_desc = fused_node->GetOpDesc()->GetInputDesc(0);
  vector<int64_t> img_input_shape = img_input_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(img_input_shape.size() != 4, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input0 shape dim must be 4"),
                    return PARAM_INVALID);
  // get input 2 shape
  ge::GeTensorDesc warpoffset_input_desc = fused_node->GetOpDesc()->GetInputDesc(1);
  vector<int64_t> warpoffset_shape = warpoffset_input_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(warpoffset_shape.size() != 4, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input1 shape dim must be 4"),
                    return PARAM_INVALID);
  // get output 1 shape
  ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> output_desc_shape = output_desc.GetShape().GetDims();
  FUSION_PASS_CHECK(output_desc_shape.size() != 4, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the output shape dim must be 4"),
                    return PARAM_INVALID);
  // uint8 input
  int64_t ret = 0;
  if (img_input_desc.GetDataType() == DT_UINT8) {
    // create pre_cast1_node
    ge::NodePtr pre_cast1_node = nullptr;
    ret = CreateCastNode("_pre_cast1_node", pre_cast1_node, fused_node, graph, newNodes, img_input_desc);
    FUSION_PASS_CHECK(pre_cast1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pre_cast1_node is null."),
                      return PARAM_INVALID);
    AttrUtils::SetInt(pre_cast1_node->GetOpDesc(), "dst_type", DT_FLOAT16);

    // create pre_cast2_node
    ge::NodePtr pre_cast2_node = nullptr;
    ret = CreateCastNode("_pre_cast2_node", pre_cast2_node, fused_node, graph, newNodes, output_desc);
    FUSION_PASS_CHECK(pre_cast2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pre_cast2_node is null."),
                      return PARAM_INVALID);
    AttrUtils::SetInt(pre_cast2_node->GetOpDesc(), "dst_type", DT_UINT8);
    // img->pre_cast1_node
    auto tmpPreDataAnchor = fused_node->GetInDataAnchor(0)->GetPeerOutAnchor();
    ret =
        ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(), fused_node->GetInDataAnchor(0));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge img_node failed."), return FAILED);
    ret = ge::GraphUtils::AddEdge(tmpPreDataAnchor, pre_cast1_node->GetInDataAnchor(0));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge img to pre_cast1_node failed."),
                      return FAILED);

    // pre_cast1_node->fused_node
    ret = ge::GraphUtils::AddEdge(pre_cast1_node->GetOutDataAnchor(0), fused_node->GetInDataAnchor(0));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge pre_cast1_node to fused_node failed."),
                      return FAILED);

    // pre_cast2_node-> fused_node output
    for (auto inDataAnchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return NOT_CHANGED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(pre_cast2_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge failed."), return NOT_CHANGED);
    }
    // fused_node->pre_cast2_node
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fused_node->GetOutDataAnchor(0),
                                              pre_cast2_node->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge failed."), return NOT_CHANGED);
    img_input_desc.SetDataType(DT_FLOAT16);
    ge::OpDescPtr fusedDesc = fused_node->GetOpDesc();
    fusedDesc->UpdateInputDesc("image", img_input_desc);
    output_desc.SetDataType(DT_FLOAT16);
    fusedDesc->UpdateOutputDesc("map_img", output_desc);
  }

  ge::GeTensorDesc temp_desc = warpoffset_input_desc;
  vector<int64_t> temp_shape = warpoffset_shape;
  temp_shape[1] = 1;
  temp_shape[2] = warpoffset_shape[1];
  temp_shape[3] = warpoffset_shape[2];
  ge::GeShape new_shape(temp_shape);  // [N1HW]
  temp_desc.SetShape(new_shape);
  temp_desc.SetOriginShape(new_shape);

  // create split node
  ge::NodePtr split_node = nullptr;
  ret = CreateSplitNode(split_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(split_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "split_node is null."), return PARAM_INVALID);

  // create floor_x node
  ge::NodePtr floor_x_node = nullptr;
  ret = CreateFloorxNode(floor_x_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(floor_x_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "floor_x_node is null."),
                    return PARAM_INVALID);

  // create ceil_x node
  ge::NodePtr ceil_x_node = nullptr;
  ret = CreateCeilxNode(ceil_x_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(ceil_x_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ceil_x_node is null."),
                    return PARAM_INVALID);

  // create floor_y node
  ge::NodePtr floor_y_node = nullptr;
  ret = CreateFlooryNode(floor_y_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(floor_y_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "floor_y_node is null."),
                    return PARAM_INVALID);

  // create ceil_y node
  ge::NodePtr ceil_y_node = nullptr;
  ret = CreateCeilyNode(ceil_y_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(ceil_y_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ceil_y_node is null."),
                    return PARAM_INVALID);

  // create cast1 node
  ge::NodePtr cast1_node = nullptr;
  ret = CreateCastNode("cast1", cast1_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(cast1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast1_node is null."), return PARAM_INVALID);

  // create cast2 node
  ge::NodePtr cast2_node = nullptr;
  ret = CreateCastNode("cast2", cast2_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(cast2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast1_cast2_nodenode is null."),
                    return PARAM_INVALID);

  // create cast3 node
  ge::NodePtr cast3_node = nullptr;
  ret = CreateCastNode("cast3", cast3_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(cast3_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast3_node is null."), return PARAM_INVALID);

  // create cast4 node
  ge::NodePtr cast4_node = nullptr;
  ret = CreateCastNode("cast4", cast4_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(cast4_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast4_node is null."), return PARAM_INVALID);

  // create mulsx1 node
  ge::NodePtr mulsx1_node = nullptr;
  float mul_x_factor = img_input_shape[3];
  ret = CreateMulsx1Node(mulsx1_node, fused_node, graph, newNodes, temp_desc, mul_x_factor);
  FUSION_PASS_CHECK(mulsx1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulsx1_node is null."),
                    return PARAM_INVALID);

  // create mulsx2 node
  ge::NodePtr mulsx2_node = nullptr;
  ret = CreateMulsx2Node(mulsx2_node, fused_node, graph, newNodes, temp_desc, mul_x_factor);
  FUSION_PASS_CHECK(mulsx2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulsx2_node is null."),
                    return PARAM_INVALID);

  // create mulsy1 node
  float mul_y_factor = img_input_shape[2] * img_input_shape[3];
  ge::NodePtr mulsy1_node = nullptr;
  ret = CreateMulsy1Node(mulsy1_node, fused_node, graph, newNodes, temp_desc, mul_y_factor);
  FUSION_PASS_CHECK(mulsy1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulsy1_node is null."),
                    return PARAM_INVALID);

  // create mulsy2 node
  ge::NodePtr mulsy2_node = nullptr;
  ret = CreateMulsy2Node(mulsy2_node, fused_node, graph, newNodes, temp_desc, mul_y_factor);
  FUSION_PASS_CHECK(mulsy2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulsy2_node is null."),
                    return PARAM_INVALID);

  std::string name;
  // create add1 node
  ge::NodePtr add1_node = nullptr;
  name = "add1";
  ret = CreateAddNode(name, add1_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(add1_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add1_node is null."), return PARAM_INVALID);

  // create add2 node
  ge::NodePtr add2_node = nullptr;
  name = "add2";
  ret = CreateAddNode(name, add2_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(add2_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add2_node is null."), return PARAM_INVALID);

  // create add3 node
  ge::NodePtr add3_node = nullptr;
  name = "add3";
  ret = CreateAddNode(name, add3_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(add3_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add3_node is null."), return PARAM_INVALID);

  // create add4 node
  ge::NodePtr add4_node = nullptr;
  name = "add4";
  ret = CreateAddNode(name, add4_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(add4_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add4_node is null."), return PARAM_INVALID);

  // create concat node
  ge::NodePtr concat_node = nullptr;
  ret = CreateConcatNode(concat_node, fused_node, graph, newNodes, temp_desc);
  FUSION_PASS_CHECK(concat_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concat_node is null."),
                    return PARAM_INVALID);

  // create a new node for remapOffsets
  ge::NodePtr remap_offsets_node = nullptr;
  ret = CreateRemapOffsetsNode(remap_offsets_node, fused_node, graph, newNodes, img_input_desc, warpoffset_input_desc);
  FUSION_PASS_CHECK(remap_offsets_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remap_offsets_node is null."),
                    return PARAM_INVALID);

  // create a new node for Resize
  ge::NodePtr remap_resize_node = nullptr;
  ret = CreateRemapResizeNode(remap_resize_node, fused_node, graph, newNodes, img_input_desc, warpoffset_input_desc,
                              output_desc);
  FUSION_PASS_CHECK(remap_resize_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remap_resize_node is null."),
                    return PARAM_INVALID);

  // warp_offset->split_node & warp_offset->resize_node
  auto tmpDataAnchor = fused_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  ret = ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(1)->GetPeerOutAnchor(), fused_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge offsets_node failed."), return FAILED);

  ret = ge::GraphUtils::AddEdge(tmpDataAnchor, remap_resize_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge offsets_node to resize node failed."),
                    return FAILED);

  // split_node->floor_x_node
  ret = ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), floor_x_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge split_node to floor_x_node failed."),
                    return FAILED);

  // split_node->ceil_x_node
  ret = ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(0), ceil_x_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge split_node to ceil_x_node failed."),
                    return FAILED);

  // split_node->floor_y_node
  ret = ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), floor_y_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge split_node to floor_y_node failed."),
                    return FAILED);

  // split_node->floor_y_node
  ret = ge::GraphUtils::AddEdge(split_node->GetOutDataAnchor(1), ceil_y_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge split_node to ceil_y_node failed."),
                    return FAILED);

  // floor_x_node->cast1_node
  ret = ge::GraphUtils::AddEdge(floor_x_node->GetOutDataAnchor(0), cast1_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge floor_x_node to cast1_node failed."),
                    return FAILED);

  // ceil_x_node->cast2_node
  ret = ge::GraphUtils::AddEdge(ceil_x_node->GetOutDataAnchor(0), cast2_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge ceil_x_node to cast2_node failed."),
                    return FAILED);

  // floor_y_node->cast3_node
  ret = ge::GraphUtils::AddEdge(floor_y_node->GetOutDataAnchor(0), cast3_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge floor_y_node to cast3_node failed."),
                    return FAILED);

  // ceil_y_node->cast4_node
  ret = ge::GraphUtils::AddEdge(ceil_y_node->GetOutDataAnchor(0), cast4_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge ceil_y_node to cast4_node failed."),
                    return FAILED);

  // cast1_node->mulsx1_node
  ret = ge::GraphUtils::AddEdge(cast1_node->GetOutDataAnchor(0), mulsx1_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge cast1_node to mulsx1_node failed."),
                    return FAILED);

  // cast2_node->mulsx2_node
  ret = ge::GraphUtils::AddEdge(cast2_node->GetOutDataAnchor(0), mulsx2_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge cast2_node to mulsx2_node failed."),
                    return FAILED);

  // cast3_node->mulsy1_node
  ret = ge::GraphUtils::AddEdge(cast3_node->GetOutDataAnchor(0), mulsy1_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge cast3_node to mulsy1_node failed."),
                    return FAILED);

  // cast4_node->mulsy2_node
  ret = ge::GraphUtils::AddEdge(cast4_node->GetOutDataAnchor(0), mulsy2_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge cast4_node to mulsy2_node failed."),
                    return FAILED);

  // mulsx1_node + mulsy1_node->add1_node
  ret = ge::GraphUtils::AddEdge(mulsx1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsx1_node to add1_node failed."),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(mulsy1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsy1_node to add1_node failed."),
                    return FAILED);

  // mulsx2_node + mulsy1_node ->add2_node
  ret = ge::GraphUtils::AddEdge(mulsx2_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsx2_node to concat_node failed."),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(mulsy1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsy1_node to add1_node failed."),
                    return FAILED);

  // mulsx1_node + mulsy2_node->add3_node
  ret = ge::GraphUtils::AddEdge(mulsx1_node->GetOutDataAnchor(0), add3_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsx1_node to add1_node failed."),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(mulsy2_node->GetOutDataAnchor(0), add3_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsy2_node to concat_node failed."),
                    return FAILED);

  // mulsx2_node + mulsy2_node->add4_node
  ret = ge::GraphUtils::AddEdge(mulsx2_node->GetOutDataAnchor(0), add4_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsx2_node to concat_node failed."),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(mulsy2_node->GetOutDataAnchor(0), add4_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsy2_node to concat_node failed."),
                    return FAILED);

  // add1_node->concat_node
  ret = ge::GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add1_node to concat_node failed."),
                    return FAILED);

  // add2_node->concat_node
  ret = ge::GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add2_node to concat_node failed."),
                    return FAILED);

  // add3_node->concat_node
  ret = ge::GraphUtils::AddEdge(add3_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add3_node to concat_node failed."),
                    return FAILED);

  // add4_node->concat_node
  ret = ge::GraphUtils::AddEdge(add4_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add4_node to concat_node failed."),
                    return FAILED);

  // concat_node->remap_offsets_node
  ret = ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), remap_offsets_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge mulsy2_node to concat_node failed."),
                    return FAILED);

  // img->remap_offsets_node
  auto tempDataAnchor = fused_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  ret = ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(), fused_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge img_node failed."), return FAILED);
  ret = ge::GraphUtils::AddEdge(tempDataAnchor, remap_offsets_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge img to remap_offsets_node failed."),
                    return FAILED);

  // remap_offsets_node->remap_resize_node
  ret = ge::GraphUtils::AddEdge(remap_offsets_node->GetOutDataAnchor(0), remap_resize_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge offsets_node to resize_node failed."),
                    return FAILED);

  // remap_resize_node->warp_img
  for (auto inDataAnchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return NOT_CHANGED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(remap_resize_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge failed."), return NOT_CHANGED);
  }
  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != GRAPH_SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return NOT_CHANGED);
  // insert transpose at resize input 1
  vector<int64_t> perm_boxes_list = {0, 3, 1, 2};
  AddTransposeBeforeNode(remap_resize_node, 1, perm_boxes_list, graph);
  ret = ge::GraphUtils::AddEdge(remap_resize_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                split_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge transpose to split node failed."),
                    return FAILED);
  // insert transpose at resize input 0
  perm_boxes_list = {0, 1, 4, 2, 3};
  AddTransposeBeforeNode(remap_resize_node, 0, perm_boxes_list, graph);
  // insert transpose at remap_offsets output 0
  perm_boxes_list = {0, 2, 3, 1};
  AddTransposeAfterNode(remap_resize_node, 0, perm_boxes_list, graph);

  return SUCCESS;
}

REGISTER_PASS("RemapFusionPass", BUILT_IN_GRAPH_PASS, RemapFusionPass);
}  // namespace fe
