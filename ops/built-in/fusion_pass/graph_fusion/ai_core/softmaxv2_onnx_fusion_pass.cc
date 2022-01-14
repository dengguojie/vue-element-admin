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

/*
    Softmaxv2---->         data
                            |
                            |
                          reshape1
                            |
                            |
                          softmaxv2
                            |
                          reshape2
*/

#include "softmaxv2_onnx_fusion_pass.h"
#include <vector>
#include <string>

#include "external/graph/operator_factory.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SoftmaxV2";
static const vector<vector<int64_t>> SHAPES = {{1000, 5, 64, 64},
                                              {1000, 12, 48, 48}};
static const std::string PATTERN_FUSEDNODE = "SoftmaxV2";
vector<FusionPattern*> ASoftmaxFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ASoftmaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern object failed."),
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
  std::vector<int64_t> dims = softmax_opdesc->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(softmax_opdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmax_opdesc is null, fusion failed."),
                    return PARAM_INVALID);

  // helper 项目白名单
  for (auto SHAPE : SHAPES) {
    if (dims == SHAPE) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ASoftmaxFusionPass is not support.");
      return NOT_CHANGED;
    }
  }

  // only softmax-11 and x.dim > 2 need insert reshape node
  int need_fusion = 0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(softmax_opdesc, "need_fusion", need_fusion) || (dims.size() < 3),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get ATTR need_fusion fail or dims.size() < 3"),
                    return NOT_CHANGED);

  // reshape input.shape = [x0, ..., xk, ..., xn] to [x0*...*x(k-1), xk*...*xn]=[dim_1, dim_2]
  ge::NodePtr reshape1_node = nullptr;
  ge::NodePtr shape1_node = nullptr;
  std::vector<int32_t> axes;
  ge::AttrUtils::GetListInt(softmax_opdesc, "axes", axes);
  unsigned int axis = axes[0] > -1 ? axes[0] : axes[0] + dims.size();
  int64_t dim_1 = 1;
  int64_t dim_2 = 1;
  for (size_t idx = 0; idx < dims.size(); idx++) {
    if (idx < axis) {
      dim_1 *= dims[idx];
    } else {
      dim_2 *= dims[idx];
    }
  }
  FUSION_PASS_CHECK(CreateReshapeNode(graph, softmax_node, reshape1_node, shape1_node, {dim_1, dim_2}) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode FAILED."),
                    return PARAM_INVALID);

  // reshape softmax.shape = [dim_1, dim_2] to origin shape
  ge::NodePtr reshape2_node = nullptr;
  ge::NodePtr shape2_node = nullptr;
  FUSION_PASS_CHECK(CreateReshapeNode(graph, reshape1_node, reshape2_node, shape2_node, dims) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode FAILED."),
                    return PARAM_INVALID);

  // update softmax's tensordesc info
  auto input_new_desc = reshape1_node->GetOpDesc()->GetOutputDesc(0);
  auto ouput_new_desc = reshape2_node->GetOpDesc()->GetInputDesc(0);
  softmax_opdesc->UpdateInputDesc(0, input_new_desc);
  softmax_opdesc->UpdateOutputDesc(0, ouput_new_desc);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(softmax_opdesc, "axes", {1}),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr axes failed"), return PARAM_INVALID);

  // handle edge
  auto out_anchor = softmax_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(out_anchor, softmax_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove edge from fused first input FAILED."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(out_anchor, reshape1_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from fused first input FAILED."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshape1_node->GetOutDataAnchor(0),
                                            softmax_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Fail to add edge between reshape1 and softmax."),
                    return PARAM_INVALID);

  auto in_anchors = softmax_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  for (auto in_anchor : in_anchors) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(softmax_node->GetOutDataAnchor(0), in_anchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "remove edge from fused out to other FAILED."),
                    return PARAM_INVALID);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshape2_node->GetOutDataAnchor(0), in_anchor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge from reshape to other FAILED."),
                    return PARAM_INVALID);
  }

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmax_node->GetOutDataAnchor(0),
                                            reshape2_node->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "add edge from fused to reshape FAILED."),
                    return PARAM_INVALID);

  return SUCCESS;
}

Status ASoftmaxFusionPass::CreateReshapeNode(ge::ComputeGraph& graph,
                                             ge::NodePtr& anchor_node,
                                             ge::NodePtr& reshape_node,
                                             ge::NodePtr& const_node,
                                             vector<int64_t> dims) {
  auto anchor_desc = anchor_node->GetOpDesc();
  ge::Operator reshape_op;
  ge::GeTensorDesc x_desc;
  ge::GeTensorDesc y_desc;
  if (dims.size() == 2) {
    // the first reshape
    reshape_op = ge::OperatorFactory::CreateOperator("Reshape_" + anchor_desc->GetName(), "Reshape");
    x_desc = anchor_node->GetOpDesc()->GetInputDesc(0);
    y_desc = anchor_node->GetOpDesc()->GetOutputDesc(0);
    y_desc.SetShape(ge::GeShape(dims));
    y_desc.SetOriginShape(ge::GeShape(dims));
  } else {
    // the second reshape
    reshape_op = ge::OperatorFactory::CreateOperator(anchor_desc->GetName() + "_reshape", "Reshape");
    x_desc = anchor_node->GetOpDesc()->GetOutputDesc(0);
    y_desc = anchor_node->GetOpDesc()->GetInputDesc(0);
  }

  FUSION_PASS_CHECK(reshape_op.IsEmpty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "create fusion node %s failed", FUSED_OP_TYPE.c_str()),
                    return FAILED);

  auto reshapeOpDescPtr = ge::OpDescUtils::GetOpDescFromOperator(reshape_op);

  reshapeOpDescPtr->UpdateInputDesc("x", x_desc);
  reshapeOpDescPtr->UpdateOutputDesc("y", y_desc);

  ge::GeTensorDesc shape_desc = x_desc.Clone();
  shape_desc.SetDataType(ge::DT_INT32);
  shape_desc.SetOriginDataType(ge::DT_INT32);
  shape_desc.SetShape(ge::GeShape({(int64_t)dims.size()}));
  shape_desc.SetOriginShape(ge::GeShape({(int64_t)dims.size()}));
  reshapeOpDescPtr->UpdateInputDesc("shape", shape_desc);
  ge::GeTensorPtr shape_tensor_ptr = nullptr;
  FUSION_PASS_MAKE_SHARED((shape_tensor_ptr = std::make_shared<ge::GeTensor>(
                               shape_desc, reinterpret_cast<uint8_t *>(dims.data()), dims.size() * sizeof(int64_t))),
                          return FAILED);
  ge::OpDescPtr shapeDesc = ge::OpDescUtils::CreateConstOp(shape_tensor_ptr);

  reshape_node = graph.AddNode(reshapeOpDescPtr);
  FUSION_PASS_CHECK(reshape_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode reshape_node is nullptr"),
                    return FAILED);

  const_node = graph.AddNode(shapeDesc);
  FUSION_PASS_CHECK(const_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CreateReshapeNode const_node is nullptr"),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                                            reshape_node->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add edge between reshape and const"),
                    return FAILED);
  return SUCCESS;
}

REGISTER_PASS("ASoftmaxFusionPass", BUILT_IN_GRAPH_PASS, ASoftmaxFusionPass);
}  // fe