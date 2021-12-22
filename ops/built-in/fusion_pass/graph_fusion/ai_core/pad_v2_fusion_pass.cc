/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file pad_v2_fusion_pass.cpp
 * \brief split fusion pass(padv2 --> pad_v2_d)
 */
#include "pad_v2_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"


using namespace ge;
namespace fe {
static const std::string PATTERN_PADV2 = "PadV2";
static const char* PADV2 = "PadV2";

bool PadV2FusionPass::GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                                    std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGW(op.GetName().c_str(), "Get const data failed."),
                      return false);
    if (const_data_ptr == nullptr) {
      VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "const_data_ptr is null");
    }
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    FUSION_PASS_CHECK(const_data_ptr == nullptr, OP_LOGW(op.GetName().c_str(), "Get const data failed."),
                      return false);
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int64 fusion pass ====== %d", (int64_t)(*(const_data_ptr + i)));
    }
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

vector<FusionPattern*> PadV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // pad fusion to pad_d
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADV2, {PADV2}).SetOutput(PATTERN_PADV2);

  patterns.push_back(pattern);

  return patterns;
}

Status PadV2FusionPass::PadMoveConsttoAttr(ge::ComputeGraph& graph, ge::NodePtr& pad_node, const string& attr_name,
                                           int32_t index) {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  vector<string> dummyVec;
  op_desc->SetOpInferDepends(dummyVec);
  Tensor const_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", const_tensor)) {
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("paddings").GetDataType();

  std::vector<int64_t> pad_value;
  if (!GetConstValue(op, const_tensor, dtype, pad_value)) {
    return GRAPH_FAILED;
    VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "Get Const Value failed ");
  };

  vector<vector<int64_t>> paddings;
  for (size_t i = 1; i < pad_value.size(); i += 2) {
    vector<int64_t> one_value;
    one_value.push_back(pad_value[i - 1]);
    one_value.push_back(pad_value[i]);
    paddings.push_back(one_value);
  }

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::AttrUtils::SetListListInt(pad_desc, attr_name, paddings);

  // get pad const achor
  ge::InDataAnchorPtr pad_anchor_ptr1 = pad_node->GetInDataAnchor(index);
  ge::OutDataAnchorPtr const_anchor_ptr = pad_anchor_ptr1->GetPeerOutAnchor();
  ge::NodePtr constNode1 = const_anchor_ptr->GetOwnerNode();

  // delete const input node, edge
  ge::GraphUtils::RemoveEdge(const_anchor_ptr, pad_anchor_ptr1);
  ge::NodeUtils::ClearInDataAnchor(pad_node, pad_anchor_ptr1);
  ge::OpDescUtils::ClearInputDesc(pad_desc, index);
  if (PatternFusionUtil::GetOutEdgeSize(constNode1) == 0) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode1),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed",
                                                     constNode1->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode1->GetName().c_str());
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.", constNode1->GetName().c_str());
  }
  return SUCCESS;
}

Status PadV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get pad node and node-desc
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PADV2, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  NOT_CHANGED_WITH_DYNAMIC_NODE({pad_node})

  std::vector<PassAttrInfo> attr_infos = {{1, "paddings", "SetInt"}};
  const std::string fusion_op_type = "PadV2D";
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(pad_node, fusion_op_type, attr_infos);
  if (fusionDescPtr == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr.");
    return NOT_CHANGED;
  }

  if (PadMoveConsttoAttr(graph, pad_node, "paddings", 1) != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), " PadMoveConsttoAttr failed.");
    return PARAM_INVALID;
  }

  vector<bool> is_input_const = {false};
  pad_desc->SetIsInputConst(is_input_const);
  // set op type PadV2->PadV2D
  pad_desc->SetType("PadV2D");

  // Create new node to replace "pad_node",
  // otherwise dynamic_pad_d can't find the true InferShape.
  // connect: AddEdge(src, dst) must follow 0st node's output(src) to connect 1st node's input(dst).
  ge::OpDescPtr fusionDesc = ge::AttrUtils::CopyOpDesc(pad_desc);
  auto realFusedOp = ge::OperatorFactory::CreateOperator("realFusedOp", "PadV2D");
  if (realFusedOp.IsEmpty()) {
    VECTOR_FUSION_INNER_ERR_REPORT("PadV2D", "create fusion node %s failed", "PadV2D");
    return FAILED;
  }
  auto realFusedOpDescPtr = ge::OpDescUtils::GetOpDescFromOperator(realFusedOp);
  realFusedOp.BreakConnect();
  fusionDesc->AddInferFunc(realFusedOpDescPtr->GetInferFunc());
  ge::NodePtr fusion_node = nullptr;
  fusion_node = graph.AddNode(fusionDesc);

  // replace input anchor
  ge::InDataAnchorPtr InPtr_pad = pad_node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr OutPtr_InPtr_pad = InPtr_pad->GetPeerOutAnchor();
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(OutPtr_InPtr_pad, fusion_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Add Input Edge failed."), return FAILED);

  ge::InDataAnchorPtr InPtr_pad2 = pad_node->GetInDataAnchor(1);
  ge::OutDataAnchorPtr OutPtr_InPtr_pad2 = InPtr_pad2->GetPeerOutAnchor();
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(OutPtr_InPtr_pad2, fusion_node->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Add Input Edge failed."), return FAILED);

  // replace output anchor: must remove first, then connect.
  ge::OutDataAnchorPtr OutPtr_pad = pad_node->GetOutDataAnchor(0);
  for (auto inDataAnchor : OutPtr_pad->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pad_node->GetOutDataAnchor(0), inDataAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Remove Output Edge failed."), return FAILED);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(0), inDataAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Add Output Edge failed."), return FAILED);
  }

  // replace control anchor: must remove first, then connect.
  if (pad_node->GetOutControlAnchor()) {
    for (auto inControlAnchor : pad_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pad_node->GetOutControlAnchor(), inControlAnchor),
                        VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Remove OutputControl Edge failed."),
                                                       return FAILED);

      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusion_node->GetOutControlAnchor(), inControlAnchor),
                        VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass", "Add OutputControl Edge failed."),
                                                       return FAILED);
    }
  }

  // remove org_node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(pad_node), VECTOR_FUSION_INNER_ERR_REPORT("PadV2FusionPass",
                    "Remove OrgNode failed."),
                    return FAILED);

  // push new_node
  fusionNodes.push_back(fusion_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "PadV2FusionPass SUCCESS!");

  return SUCCESS;
}

REGISTER_PASS("PadV2FusionPass", BUILT_IN_GRAPH_PASS, PadV2FusionPass);
}  // namespace fe
