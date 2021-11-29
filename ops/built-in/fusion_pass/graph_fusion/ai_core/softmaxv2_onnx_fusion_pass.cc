/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*
    Softmaxv2---->         data 
                            |
                            |
                          flatten
                            |
                            |
                          softmaxv2
                            |
                          reshape
*/

#include "softmaxv2_onnx_fusion_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SoftmaxV2";
static const std::string PATTERN_FUSEDNODE = "SoftmaxV2";
vector<FusionPattern*> ASoftmaxFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ASoftmaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status ASoftmaxFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  ge::NodePtr softmax_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(softmax_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmaxNode is null, fusion failed."),
                    return PARAM_INVALID);
  
  auto softmax_opdesc = softmax_node->GetOpDesc();
  FUSION_PASS_CHECK(softmax_opdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmax_opdesc is null, fusion failed."),
                    return PARAM_INVALID);
  
  if (!CheckIsNeedFusion(softmax_node)) {
    return NOT_CHANGED;
  }
  
  ge::NodePtr flatten_node = nullptr;
  FUSION_PASS_CHECK(CreateFlattenNode(graph, softmax_node, flatten_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateFlattenNode FAILED."),
                    return PARAM_INVALID);

  ge::NodePtr reshape_node = nullptr;
  FUSION_PASS_CHECK(CreateReshapeNode(graph, softmax_node, flatten_node, reshape_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode FAILED."),
                    return PARAM_INVALID);
  
  ge::NodePtr const_node = nullptr;
  FUSION_PASS_CHECK(CreateConstNode(graph, softmax_node, const_node) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateConstNode FAILED."),
                    return PARAM_INVALID);
  
  auto input_new_desc = flatten_node->GetOpDesc()->GetOutputDesc(0);
  auto ouput_new_desc = reshape_node->GetOpDesc()->GetInputDesc(0);
  softmax_opdesc->UpdateInputDesc(0, input_new_desc);
  softmax_opdesc->UpdateOutputDesc(0, ouput_new_desc);

  auto out_anchor = softmax_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor, softmax_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge from fused first input FAILED."),
                    return PARAM_INVALID);
  
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(out_anchor, flatten_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge from fused first input FAILED."),
                    return PARAM_INVALID);
  
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(flatten_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from flatten to softmax input FAILED."),
                    return PARAM_INVALID);
  
  auto in_anchors = softmax_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  for (auto in_anchor : in_anchors) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmax_node->GetOutDataAnchor(0), in_anchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge from fused out to other FAILED."),
                    return PARAM_INVALID);
    
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), in_anchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from reshape to other FAILED."),
                    return PARAM_INVALID);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from fused to reshape FAILED."),
                    return PARAM_INVALID);
  
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from fused to reshape FAILED."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(softmax_opdesc, "axes", {1}),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get attr axes failed"), return PARAM_INVALID);
  return SUCCESS;
}

bool ASoftmaxFusionPass::CheckIsNeedFusion(ge::NodePtr& fused_node) {
  auto opdesc = fused_node->GetOpDesc();
  int need_fusion = 0;
  if (!ge::AttrUtils::GetInt(opdesc, "need_fusion", need_fusion)) {
    OP_LOGW("ASoftmaxFusionPass", "Get ATTR need_fusion fail");
    return false;
  }
  auto dims = opdesc->GetInputDesc(0).GetShape().GetDims();
  std::vector<int32_t> axis;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(opdesc, "axes", axis),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get attr axes failed"), return false);
  if (axis.empty()) {
    return true;
  }

  int dims_size = dims.size();
  int dim = axis[0] >= 0 ? axis[0] : dims_size + axis[0];
  if (dim == 1 || dim == dims_size - 1) {
    return false;
  }
  return true;
}

Status ASoftmaxFusionPass::CreateFlattenNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& new_node) {
  auto fused_desc = fused_node->GetOpDesc();
  std::shared_ptr<ge::OpDesc> flatten_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (flatten_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_flatten", "Flatten")),
    return FAILED);

  FUSION_PASS_CHECK(flatten_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateFlattenNode node"),
                    return FAILED);
  
  std::vector<int32_t> axis;
  if (!ge::AttrUtils::GetListInt(fused_desc, "axes", axis) || axis.empty()) {
    axis.push_back(1);
  }
  
  auto input_desc = fused_desc->GetInputDesc(0);
  flatten_desc->AddInputDesc("x", input_desc);

  auto output_desc = input_desc.Clone();
  auto dims = output_desc.GetShape().GetDims();
  int dim = axis[0] >= 0 ? axis[0] : dims.size() + axis[0];
  if (dim >= dims.size()) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "attr axis %d is wron", axis[0]);
    return FAILED;
  }

  std::vector<int64_t> new_dims;
  int val = 1;
  for (int i = 0; i < dim; ++i) {
    val *= dims[i];
  }
  new_dims.push_back(val);
  val = 1;
  for (int i = dim; i < dims.size(); ++i) {
    val *= dims[i];
  }
  new_dims.push_back(val);

  output_desc.SetShape(ge::GeShape(new_dims));
  flatten_desc->AddOutputDesc("y", output_desc);
  ge::AttrUtils::SetListInt(flatten_desc, "axes", {dim});
  new_node = graph.AddNode(flatten_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateFlattenNode new_node is nullptr"),
                    return FAILED);
  return SUCCESS;
}

Status ASoftmaxFusionPass::CreateReshapeNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& flatten_node,
                                             ge::NodePtr& new_node) {
  auto fused_desc = fused_node->GetOpDesc();
  std::shared_ptr<ge::OpDesc> reshape_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (reshape_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_reshape", "Reshape")),
    return FAILED);

  FUSION_PASS_CHECK(reshape_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode reshape_desc is nullptr"),
                    return FAILED);
  auto input_desc = flatten_node->GetOpDesc()->GetOutputDesc(0);
  reshape_desc->AddInputDesc("x", input_desc);
  
  auto input_desc1 = input_desc.Clone();
  input_desc1.SetDataType(ge::DT_INT32);
  std::vector<int64_t> dims = {(int)fused_desc->GetInputDesc(0).GetShape().GetDimNum()};
  input_desc1.SetShape(ge::GeShape(dims));
  reshape_desc->AddInputDesc("shape", input_desc1);

  auto output_desc = fused_desc->GetOutputDesc(0);
  reshape_desc->AddOutputDesc("y", output_desc);
  new_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode new_node is nullptr"),
                    return FAILED); 
  return SUCCESS;                 
}

Status ASoftmaxFusionPass::CreateConstNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& new_node) {
  auto fused_desc = fused_node->GetOpDesc();
  std::shared_ptr<ge::OpDesc> const_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    (const_desc = std::make_shared<ge::OpDesc>(fused_desc->GetName() + "_const", "Const")),
    return FAILED);
  
  FUSION_PASS_CHECK(const_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create const node"),
                    return FAILED);
  auto dims = fused_desc->GetOutputDesc(0).GetShape().GetDims();
  int len = dims.size();
  ge::GeShape shape({len});
  ge::GeTensorDesc desc(shape, ge::FORMAT_ND, ge::DT_INT64);
  ge::GeTensorPtr tensor_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((tensor_ptr = std::make_shared<ge::GeTensor>(
                               desc, reinterpret_cast<uint8_t*>(dims.data()), len * sizeof(int64_t))),
                          tensor_ptr = nullptr;
                          return PARAM_INVALID);
  ge::AttrUtils::SetTensor(const_desc, "value", tensor_ptr);
  const_desc->AddOutputDesc(tensor_ptr->GetTensorDesc());
  new_node = graph.AddNode(const_desc);
  FUSION_PASS_CHECK(new_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to  CreateConstNode node"),
                    return FAILED);
  return SUCCESS;
}
REGISTER_PASS("ASoftmaxFusionPass", BUILT_IN_GRAPH_PASS, ASoftmaxFusionPass);
}  // fe