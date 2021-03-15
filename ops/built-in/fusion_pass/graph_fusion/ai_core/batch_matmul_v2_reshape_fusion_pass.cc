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
 * \file batch_matmul_v2_reshape_fusion_pass.cc
 * \brief the pass will effect if the input shape of batch_matmul_v2 is 1D
 */

#include "batch_matmul_v2_reshape_fusion_pass.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
  static const string PATTERN_BATCHMATMULV2 = "BatchMatMulV2";
  static const string BATCHMATMULV2 = "BatchMatMulV2";
  vector<FusionPattern*> BatchMatMulV2ReshapeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns; 
  FusionPattern *pattern = new(std::nothrow) FusionPattern("BatchMatMulV2ReshapeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object fail."), 
                    return patterns);
  pattern->AddOpDesc(PATTERN_BATCHMATMULV2, {BATCHMATMULV2}).SetOutput(PATTERN_BATCHMATMULV2);
  patterns.push_back(pattern);
  return patterns;
  }

Status BatchMatMulV2ReshapeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, 
                                              vector<ge::NodePtr>& fuion_nodes) {
  OP_LOGD(FUSED_OP_TYPE, "Enter BatchMatMulV2ReshapeFusionPass.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_BATCHMATMULV2, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE, "Fuse node is null, fusion failed."), 
                    return PARAM_INVALID);
  auto x1_shape = fused_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  auto x2_shape = fused_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  if (x1_shape.size() != 1 and x2_shape.size() != 1) {
      OP_LOGD(FUSED_OP_TYPE, "Dim size of x1 or x2 for %s is not 1, graph not changed.", fused_node->GetName().c_str());
      return NOT_CHANGED;
  }
  // step1: creat reshape op and add to graph for x1
  if (x1_shape.size() == 1) {
      ge::NodePtr x1_reshape_node = nullptr;
      vector<int64_t> new_shape = {1, x1_shape[0]};
      auto in_anchor = fused_node->GetInDataAnchor(0);
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      CreateReshapeNode(graph, out_anchor, new_shape, x1_reshape_node);
      fused_node->GetOpDesc()->MutableInputDesc(0)->SetShape(ge::GeShape(new_shape));
      fused_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(ge::GeShape(new_shape));
      Status ret = InsertNode(out_anchor, in_anchor, x1_reshape_node);
      if (ret != SUCCESS) {
        OP_LOGE(x1_reshape_node->GetType().c_str(), "Add node %s failed.", x1_reshape_node->GetName().c_str());
        return FAILED;
      }
  }
  // step2: creat reshape op and add to graph for x2
  if (x2_shape.size() == 1) {
      ge::NodePtr x2_reshape_node = nullptr;
      vector<int64_t> new_shape = {x2_shape[0], 1};
      auto in_anchor = fused_node->GetInDataAnchor(1);
      auto out_anchor = in_anchor->GetPeerOutAnchor();
      CreateReshapeNode(graph, out_anchor, new_shape, x2_reshape_node);
      fused_node->GetOpDesc()->MutableInputDesc(1)->SetShape(ge::GeShape(new_shape));
      fused_node->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(ge::GeShape(new_shape));
      Status ret = InsertNode(out_anchor, in_anchor, x2_reshape_node);
      if (ret != SUCCESS) {
        OP_LOGE(x2_reshape_node->GetType().c_str(), "Add node %s failed.", x2_reshape_node->GetName().c_str());
        return FAILED;
      }
  }
  // step3: create reshape op and add to graph for batch matmul
  auto out_shape = fused_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  if (x1_shape.size() == 1) {
    out_shape.erase(out_shape.begin());
  }
  if (x2_shape.size() == 1 and out_shape.size() > 1) {
    out_shape.erase(out_shape.end()-1);
  }

  ge::NodePtr out_reshape_node = nullptr;
  auto out_anchor = fused_node->GetOutDataAnchor(0);
  CreateReshapeNode(graph, out_anchor, out_shape, out_reshape_node);
  for (const auto &peer_input_anchor: out_anchor->GetPeerAnchors()) {
      auto next_node = peer_input_anchor->GetOwnerNode();
      int idx = peer_input_anchor->GetIdx();
      next_node->GetOpDesc()->MutableInputDesc(idx)->SetShape(ge::GeShape(out_shape));
      next_node->GetOpDesc()->MutableInputDesc(idx)->SetOriginShape(ge::GeShape(out_shape));
      auto in_anchor = next_node->GetInDataAnchor(idx);
      Status ret = InsertNode(in_anchor->GetPeerOutAnchor(), in_anchor, out_reshape_node);
      if (ret != SUCCESS) {
        OP_LOGE(out_reshape_node->GetType().c_str(), "Add node %s failed.", out_reshape_node->GetName().c_str());
        return FAILED;
      }
  }
  OP_LOGD(FUSED_OP_TYPE, "BatchMatMulV2ReshapeFusionPass success.");
  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::InsertNode(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst, 
                                                 ge::NodePtr& new_node) {
  ge::NodePtr src_node = src->GetOwnerNode();
  ge::NodePtr dst_node = dst->GetOwnerNode();
  if(new_node->GetOpDesc()->UpdateInputDesc(0, src_node->GetOpDesc()->GetOutputDesc(src->GetIdx())) != SUCCESS) {
      OP_LOGI(new_node->GetName().c_str(), "update input_desc failed.");
      return FAILED;
  }
  if(new_node->GetOpDesc()->UpdateOutputDesc(0, dst_node->GetOpDesc()->GetInputDesc(dst->GetIdx())) != SUCCESS) {
    OP_LOGI(new_node->GetName().c_str(), "update output_desc failed.");
    return FAILED;
  }
  if(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS) {
    OP_LOGE(dst_node->GetName().c_str(), "Remove input0 edge error.");
    return FAILED; 
  }
  if(ge::GraphUtils::AddEdge(src, new_node->GetInDataAnchor(0)) != SUCCESS) {
    OP_LOGE(src_node->GetName().c_str(), "Add edge to node %s failed.", new_node->GetName().c_str());
    return FAILED; 
  }
  if(ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), dst)!= SUCCESS) {
   OP_LOGE(new_node->GetName().c_str(), "Add edge to node %s failed.", dst_node->GetName().c_str());
    return FAILED; 
  }
  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::CreateReshapeNode(ge::ComputeGraph& graph, const ge::OutDataAnchorPtr & out_anchor,
                                                        const vector<int64_t> & shape, ge::NodePtr& shape_node) {
  auto previous_node = out_anchor->GetOwnerNode();
  int idx = out_anchor->GetIdx();
  auto previous_node_desc = previous_node->GetOpDesc()->GetOutputDesc(idx);
  ge::GeTensorDesc next_in_desc = previous_node_desc.Clone();
  next_in_desc.SetShape(ge::GeShape(shape));
  next_in_desc.SetOriginShape(ge::GeShape(shape));

  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                          previous_node->GetName() + "/Reshape", "Reshape")), return FAILED);
  reshape_desc->AddInputDesc("x", previous_node_desc);
  reshape_desc->AddOutputDesc("y", next_in_desc);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);
  shape_node = graph.AddNode(reshape_desc);
  return SUCCESS;
}
 REGISTER_PASS("BatchMatMulV2ReshapeFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulV2ReshapeFusionPass);
}  // namespace fe

