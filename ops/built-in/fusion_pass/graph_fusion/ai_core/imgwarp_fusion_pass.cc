/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file imgwarp_fusion_pass.cpp
 * \brief imgwarp fusion pass
 */
#include "imgwarp_fusion_pass.h"
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
static const string PATTERN_FUSEDNODE = "FusedNodeIMGWarp";
static const string FUSED_NODE = "IMGWarp";

vector<FusionPattern*> IMGWarpFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("IMGWarpFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

int64_t IMGWarpFusionPass::GetDimNum(const vector<int64_t>& shapes) const {
  auto shape_lens = shapes.size();
  int64_t dim_num = 1;
  for (size_t i = 0; i < shape_lens; i++) {
    dim_num = dim_num * shapes[i];
  }
  return dim_num;
}

void IMGWarpFusionPass::SetAssitValue(float* data, const std::vector<int64_t>& shape) const {
  int64_t width = shape[3];
  int64_t height = shape[2];
  int64_t data_size = width * height;
  for (int64_t i = 0; i < data_size; i++) {
    data[i] = static_cast<float>(i / width);
  }
  for (int64_t j = 0; j < data_size; j++) {
    data[j + data_size] = static_cast<float>(j % height);
  }
}

Status IMGWarpFusionPass::CreateConstNode(vector<int64_t>& assit_tensor_shape, ge::NodePtr& fuse_node,
                                          ge::ComputeGraph& graph, ge::NodePtr& const_node) const {
  int64_t assit_size = GetDimNum(assit_tensor_shape);
  ge::GeTensorPtr assit_ptr{nullptr};
  unique_ptr<float[]> const_assit(new (std::nothrow) float[assit_size]());
  FUSION_PASS_CHECK(const_assit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_assit is NULL."),
                    return PARAM_INVALID);
  Status ret = NnSet(assit_size, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(const_assit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
  float* data_ptr = const_assit.get();
  SetAssitValue(data_ptr, assit_tensor_shape);

  ge::GeShape const_shape(assit_tensor_shape);
  ge::GeTensorDesc const_tensor_desc(const_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  FUSION_PASS_MAKE_SHARED((assit_ptr = std::make_shared<ge::GeTensor>(
                               const_tensor_desc, reinterpret_cast<uint8_t*>(data_ptr), assit_size * sizeof(float))),
                          return NOT_CHANGED);
  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assit_ptr);
  FUSION_PASS_CHECK(const_opdesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Create const op desc failed."),
                    return NOT_CHANGED);

  const_node = graph.AddNode(const_opdesc);

  return SUCCESS;
}

Status IMGWarpFusionPass::CreateAddNode(ge::NodePtr& add_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                        vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& add_input0_desc) const {
  std::shared_ptr<ge::OpDesc> add_desc = nullptr;
  std::string add_desc_name = fused_node->GetName() + "_add";
  add_desc = std::make_shared<ge::OpDesc>(add_desc_name, "Add");
  FUSION_PASS_CHECK(add_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add_desc after valid num is null, fusion failed."),
                    return NOT_CHANGED);
  // add input tensorDesc
  FUSION_PASS_CHECK(add_desc->AddInputDesc("x1", add_input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for add fail, fusion failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(add_desc->AddInputDesc("x2", add_input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input y for add fail, fusion failed."), return NOT_CHANGED);

  // add output tensorDesc
  FUSION_PASS_CHECK(add_desc->AddOutputDesc("y", add_input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add ouput z for add after fail, fusion failed."),
                    return NOT_CHANGED);

  add_node = graph.AddNode(add_desc);
  new_nodes.push_back(add_node);

  return SUCCESS;
}

Status IMGWarpFusionPass::CreateIMGWarpOffsetsNode(ge::NodePtr& offsets_node, ge::NodePtr& fused_node,
                                                   ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                                   ge::GeTensorDesc& input0_desc, ge::GeTensorDesc& input1_desc) const {
  std::shared_ptr<ge::OpDesc> offsets_desc = nullptr;
  std::string offsets_desc_name = fused_node->GetName() + "_IMGWarpOffsets";
  offsets_desc = std::make_shared<ge::OpDesc>(offsets_desc_name, "IMGWarpOffsets");
  FUSION_PASS_CHECK(offsets_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "IMGWarpOffsets_desc is null, fusion failed."), return NOT_CHANGED);
  FUSION_PASS_CHECK(offsets_desc->AddInputDesc("x", input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for offsets_node fail, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(offsets_desc->AddInputDesc("indexs", input1_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input y for offsets_node offsets_node fail, fusion failed."),
                    return NOT_CHANGED);

  // the input shape is [N, H, W, C]
  ge::GeTensorDesc output_desc = input0_desc;
  vector<int64_t> input_shape = input0_desc.GetShape().GetDims();
  // the output shape is [N, 4, C, H, W]
  vector<int64_t> output_shape = {input_shape[0], 4, input_shape[2], input_shape[3], input_shape[1]};
  ge::GeShape new_shape(output_shape);
  output_desc.SetShape(new_shape);
  output_desc.SetOriginShape(new_shape);
  FUSION_PASS_CHECK(offsets_desc->AddOutputDesc("y", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add output z for offsets_node fail, fusion failed."),
                    return NOT_CHANGED);
  offsets_node = graph.AddNode(offsets_desc);
  new_nodes.push_back(offsets_node);

  return SUCCESS;
}

Status IMGWarpFusionPass::CreateIMGWarpResizeNode(ge::NodePtr& resize_node, ge::NodePtr& fused_node,
                                                  ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes,
                                                  ge::GeTensorDesc& input0_desc, ge::GeTensorDesc& input1_desc,
                                                  ge::GeTensorDesc& output_desc) const {
  std::shared_ptr<ge::OpDesc> resize_desc = nullptr;
  std::string resize_desc_name = fused_node->GetName() + "_IMGWarpResize";
  resize_desc = std::make_shared<ge::OpDesc>(resize_desc_name, "IMGWarpResize");
  FUSION_PASS_CHECK(resize_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "IMGWarpOffsets_desc is null, fusion failed."), return NOT_CHANGED);
  // output shape is [N,4,C,H,W]
  ge::GeTensorDesc resize_input0_desc = input0_desc;
  vector<int64_t> input0_shape = resize_input0_desc.GetShape().GetDims();
  input0_shape.insert(input0_shape.begin() + 1, 4);
  ge::GeShape new_shape(input0_shape);
  resize_input0_desc.SetShape(new_shape);
  resize_input0_desc.SetOriginShape(new_shape);
  FUSION_PASS_CHECK(resize_desc->AddInputDesc("img", resize_input0_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input x for resize_node fail, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(resize_desc->AddInputDesc("warp_offset", input1_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add input y for resize_node offsets_node fail, fusion failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(resize_desc->AddOutputDesc("warp_img", output_desc) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "add output z for resize_node fail, fusion failed."),
                    return NOT_CHANGED);
  resize_node = graph.AddNode(resize_desc);
  new_nodes.push_back(resize_node);

  return SUCCESS;
}

Status IMGWarpFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define IMGWarpFusionPass fusion begin.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "get fused_node failed, fusion failed."),
                    return NOT_CHANGED);

  // get input 2 shape
  ge::GeTensorDesc warpoffset_input_desc = fused_node->GetOpDesc()->GetInputDesc(1);
  vector<int64_t> warpoffset_shape = warpoffset_input_desc.GetShape().GetDims();

  // new a const_node
  ge::NodePtr const_node = nullptr;
  int64_t ret = 0;
  ret = CreateConstNode(warpoffset_shape, fused_node, graph, const_node);
  FUSION_PASS_CHECK(const_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const_node is null."), return PARAM_INVALID);

  // create a new node for Add
  ge::NodePtr add_node = nullptr;
  ret = CreateAddNode(add_node, fused_node, graph, newNodes, warpoffset_input_desc);
  FUSION_PASS_CHECK(add_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add_node is null."), return PARAM_INVALID);

  // create a new node for IMGWarpOffsets
  ge::NodePtr imgwarp_offsets_node = nullptr;
  ge::GeTensorDesc img_input_desc = fused_node->GetOpDesc()->GetInputDesc(0);
  ret = CreateIMGWarpOffsetsNode(imgwarp_offsets_node, fused_node, graph, newNodes, img_input_desc,
                                 warpoffset_input_desc);
  FUSION_PASS_CHECK(imgwarp_offsets_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "imgwarp_offsets_node is null."),
                    return PARAM_INVALID);

  // create a new node for Resize
  ge::NodePtr imgwarp_resize_node = nullptr;
  ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetOutputDesc(0);
  ret = CreateIMGWarpResizeNode(imgwarp_resize_node, fused_node, graph, newNodes, img_input_desc, warpoffset_input_desc,
                                output_desc);
  FUSION_PASS_CHECK(imgwarp_resize_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "imgwarp_resize_node is null."),
                    return PARAM_INVALID);

  // warp_offset->add_node
  auto tmpDataAnchor = fused_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  ret = ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(1)->GetPeerOutAnchor(), fused_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge offsets_node failed."), return FAILED);
  ret = ge::GraphUtils::AddEdge(tmpDataAnchor, add_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge offsets_node to add node failed."),
                    return FAILED);

  // const_node->add_node
  ret = ge::GraphUtils::AddEdge(const_node->GetOutAnchor(0), add_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge const_node to add node failed."),
                    return FAILED);

  // add_node->imgwarp_offsets_node
  ret = ge::GraphUtils::AddEdge(add_node->GetOutAnchor(0), imgwarp_offsets_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add_node to imgwarp_offsets_node failed."),
                    return FAILED);

  // img->imgwarp_offsets_node
  auto tempDataAnchor = fused_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  ret = ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(), fused_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge img_node failed."), return FAILED);
  ret = ge::GraphUtils::AddEdge(tempDataAnchor, imgwarp_offsets_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge img to imgwarp_offsets_node failed."),
                    return FAILED);

  // imgwarp_offsets_node->imgwarp_resize_node
  ret = ge::GraphUtils::AddEdge(imgwarp_offsets_node->GetOutDataAnchor(0), imgwarp_resize_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge offsets_node to resize_node failed."),
                    return FAILED);

  // add_node->imgwarp_resize_node
  ret = ge::GraphUtils::AddEdge(add_node->GetOutAnchor(0), imgwarp_resize_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge add_node to imgwarp_resize_node failed."),
                    return FAILED);

  // imgwarp_resize_node->warp_img
  for (auto inDataAnchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove edge failed."), return NOT_CHANGED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(imgwarp_resize_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Add edge failed."), return NOT_CHANGED);
  }
  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != GRAPH_SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Remove Node [%s] failed", fused_node->GetName().c_str()),
                    return NOT_CHANGED);

  // insert transpose at imgwarp_offsets input0
  vector<int64_t> perm_boxes_list = {0, 2, 3, 1};
  AddTransposeBeforeNode(imgwarp_offsets_node, 0, perm_boxes_list, graph);
  // insert transpose at imgwarp_offsets input1
  AddTransposeBeforeNode(imgwarp_offsets_node, 1, perm_boxes_list, graph);
  // insert transpose at imgwarp_offsets output0
  perm_boxes_list = {0, 1, 4, 2, 3};
  AddTransposeAfterNode(imgwarp_offsets_node, 0, perm_boxes_list, graph);
  return SUCCESS;
}

REGISTER_PASS("IMGWarpFusionPass", BUILT_IN_GRAPH_PASS, IMGWarpFusionPass);
}  // namespace fe
