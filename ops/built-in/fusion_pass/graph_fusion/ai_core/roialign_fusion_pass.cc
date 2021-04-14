/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file roialign_fusion_pass.h
 * \brief  fusion pass
 *
 */

#include "roialign_fusion_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "ROIAlign";
static const std::string PATTERN_FUSEDNODE = "ROIAlign";
vector<FusionPattern*> ROIAlignFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ROIAlignFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ROIAlignFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the OpDescPtr of RNN
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fused_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  bool is_have_roin_n = fused_desc->MutableInputDesc("rois_n") != nullptr;
  if (!is_have_roin_n) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input dont have rois_n, so dont change.");
    return NOT_CHANGED;
  }

  auto input_shape = fused_desc->GetInputDesc(1).GetShape();
  auto input_shape1 = fused_desc->GetInputDesc(2).GetShape();
  if (input_shape.GetDimNum() != 2) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input rois shape dim is not 2 dont need fusion.");
    return NOT_CHANGED;
  } else if (input_shape1.GetDimNum() != 1) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input rois_n shape dim is not 1 dont need fusion.");
    return NOT_CHANGED;
  } else if (input_shape.GetDim(0) != input_shape1.GetDim(0)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "input rois and rois_n shape dim 0 not equal.");
    return NOT_CHANGED;
  }

  ge::NodePtr cast_node = nullptr;
  FUSION_PASS_CHECK(MakeCastNode(graph, fused_node, cast_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeCastNode failed."), return FAILED);

  ge::NodePtr concat_node = nullptr;
  FUSION_PASS_CHECK(MakeConcatNode(graph, fused_node, cast_node, concat_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode failed."), return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                       cast_node->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from fused node to cast node failed."), return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from cast node to concat node failed."), return FAILED);

  auto peer_out_anchor = fused_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(peer_out_anchor, fused_node->GetInDataAnchor(1)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from intput1 node failed."), return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(peer_out_anchor, concat_node->GetInDataAnchor(1)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from fused node to concat node failed."), return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), fused_node->GetInDataAnchor(1)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from concat node to fused node failed."), return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fused_node->GetInDataAnchor(2)->GetPeerOutAnchor(),
                                                          fused_node->GetInDataAnchor(2)),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge from intput2 node failed."), return FAILED);

  new_nodes.push_back(cast_node);
  new_nodes.push_back(concat_node);
  return SUCCESS;
}

Status ROIAlignFusionPass::MakeCastNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node, ge::NodePtr& new_node) {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(fused_node->GetName() + "_cast", "Cast")),
                          return INTERNAL_ERROR);

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  auto input_desc1 = fused_desc->GetInputDesc(1);
  auto input_desc2 = fused_desc->GetInputDesc(2);
  auto shape = input_desc2.GetShape();
  auto dtype = input_desc2.GetDataType();
  auto format = input_desc2.GetFormat();
  auto out_dtype = input_desc1.GetDataType();

  input_desc2.SetOriginShape(shape);
  input_desc2.SetOriginFormat(format);
  auto ret = new_desc->AddInputDesc(input_desc2);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeCastNode AddInputDesc fail."), return FAILED);

  ge::GeTensorDesc cast_output_desc(GeTensorDesc(shape, format, out_dtype));
  cast_output_desc.SetOriginShape(shape);
  cast_output_desc.SetOriginFormat(format);
  ret = new_desc->AddOutputDesc(cast_output_desc);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeCastNode AddOutputDesc fail."), return FAILED);

  new_node = graph.AddNode(new_desc);
  Operator new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  new_op.SetAttr("dst_type", (int)out_dtype);
  return SUCCESS;
}

Status ROIAlignFusionPass::MakeConcatNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node,
                                          const ge::NodePtr& broad_node, ge::NodePtr& new_node) {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(fused_node->GetName() + "_ConcatD", "ConcatD")),
                          return INTERNAL_ERROR);

  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  auto input_desc1 = fused_desc->GetInputDesc(1);
  auto shape1 = input_desc1.GetShape();
  auto format = input_desc1.GetFormat();
  auto dtype = input_desc1.GetDataType();

  auto broad_node_desc = broad_node->GetOpDesc();
  auto shape2 = broad_node_desc->GetOutputDesc(0).GetShape();
  std::vector<int64_t> dims = {shape2.GetDim(0), 1};
  ge::GeShape input_shape(dims);

  new_desc->AddDynamicInputDesc("x", 2);
  ge::GeTensorDesc concat_input_desc1(input_shape, format, dtype);
  concat_input_desc1.SetOriginShape(input_shape);
  concat_input_desc1.SetOriginFormat(format);
  auto ret = new_desc->UpdateInputDesc(0, concat_input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode UpdateInputDesc 0 fail."),
                    return FAILED);

  ge::GeTensorDesc concat_input_desc2(shape1, format, dtype);
  concat_input_desc2.SetOriginShape(shape1);
  concat_input_desc2.SetOriginFormat(format);
  ret = new_desc->UpdateInputDesc(1, concat_input_desc2);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode UpdateInputDesc 1 fail."),
                    return FAILED);
  ge::AttrUtils::SetInt(new_desc, "concat_dim", 1);
  ge::AttrUtils::SetInt(new_desc, "N", 2);

  std::vector<int64_t> out_dims = {shape1.GetDim(0), 5};
  ge::GeShape out_shape(out_dims);
  ge::GeTensorDesc output_desc(out_shape, format, dtype);
  output_desc.SetOriginShape(out_shape);
  output_desc.SetOriginFormat(format);
  ret = new_desc->AddOutputDesc(output_desc);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode AddOutputDesc fail."),
                    return FAILED);

  input_desc1.SetShape(out_shape);
  ret = fused_desc->UpdateInputDesc(1, input_desc1);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode FusedNode UpdateInputDesc fail."),
                    return FAILED);
  auto input_desc2 = fused_desc->GetInputDesc(2);
  input_desc2.SetDataType(ge::DT_INT32);
  ret = fused_desc->UpdateInputDesc(2, input_desc2);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "MakeConcatNode FusedNode UpdateInputDesc2 fail."),
                    return FAILED);
  new_node = graph.AddNode(new_desc);
  return SUCCESS;
}
REGISTER_PASS("ROIAlignFusionPass", BUILT_IN_GRAPH_PASS, ROIAlignFusionPass);
}  // namespace fe