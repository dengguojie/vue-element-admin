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
 * \file pad_fusion_pass.cpp
 * \brief split fusion pass(pad --> pad_d)
 */
#include "pad_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_PAD = "Pad";
static const char* PAD = "Pad";

bool PadFusionPass::GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                                  std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    if (const_data_ptr == nullptr) {
      OP_LOGE(op.GetName().c_str(), "const_data_ptr is null");
    }
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int64 fusion pass ====== %d", (int64_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

vector<FusionPattern*> PadFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // pad fusion to pad_d
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PAD, {PAD}).SetOutput(PATTERN_PAD);

  patterns.push_back(pattern);

  return patterns;
}

Status PadFusionPass::PadMoveConsttoAttr(ge::ComputeGraph& graph, ge::NodePtr& pad_node, const string& attr_name,
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
    OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
  };

  vector<vector<int64_t>> paddings;
  for (size_t i = 1; i < pad_value.size(); i += 2) {
    vector<int64_t> one_value;
    one_value.push_back(pad_value[i - 1]);
    one_value.push_back(pad_value[i]);
    paddings.push_back(one_value);
  }

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_node's OpDesc is null, fusion failed."),
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
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode1->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode1->GetName().c_str());
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.", constNode1->GetName().c_str());
  }
  return SUCCESS;
}

Status PadFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get pad node and node-desc
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PAD, mapping);
  NOT_CHANGED_WITH_DYNAMIC_NODE({pad_node});
  ge::InDataAnchorPtr DataAnchorPtr0 = pad_node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr constAnchorPtr0 = DataAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();
  ge::GeTensorDesc InputTensor = constNode0->GetOpDesc()->GetOutputDesc(0);
  DataType dataType = InputTensor.GetDataType();
  if (dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT &&
      dataType != ge::DT_INT32 && dataType != ge::DT_INT8 &&
      dataType != ge::DT_UINT8) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Dtype of input is not supported in PadD.");
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(pad_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  std::vector<PassAttrInfo> attr_infos = {{1, "paddings", "SetInt"}};
  const std::string fusion_op_type = "PadD";
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(pad_node, fusion_op_type, attr_infos);
  if (fusionDescPtr == nullptr) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr.");
    return NOT_CHANGED;
  }

  if (PadMoveConsttoAttr(graph, pad_node, "paddings", 1) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), " PadMoveConsttoAttr failed.");
    return NOT_CHANGED;
  }

  vector<bool> is_input_const = {false};
  pad_desc->SetIsInputConst(is_input_const);
  // set op type Pad->PadD
  pad_desc->SetType("PadD");

  // Create new node to replace "pad_node",
  // otherwise dynamic_pad_d can't find the true InferShape.
  // connect: AddEdge(src, dst) must follow 0st node's output(src) to connect 1st node's input(dst).
  ge::OpDescPtr fusionDesc = ge::AttrUtils::CopyOpDesc(pad_desc);
  auto realFusedOp = ge::OperatorFactory::CreateOperator("realFusedOp", "PadD");
  if (realFusedOp.IsEmpty()) {
    OP_LOGE("PadD", "create fusion node %s failed", "PadD");
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
                    OP_LOGE("PadFusionPass", "Add Input Edge failed."), return FAILED);

  // replace output anchor: must remove first, then connect.
  ge::OutDataAnchorPtr OutPtr_pad = pad_node->GetOutDataAnchor(0);
  for (auto inDataAnchor : OutPtr_pad->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pad_node->GetOutDataAnchor(0), inDataAnchor),
                      OP_LOGE("PadFusionPass", "Remove Output Edge failed."), return FAILED);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusion_node->GetOutDataAnchor(0), inDataAnchor),
                      OP_LOGE("PadFusionPass", "Add Output Edge failed."), return FAILED);
  }

  // replace control anchor: must remove first, then connect.
  if (pad_node->GetOutControlAnchor()) {
    for (auto inControlAnchor : pad_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(pad_node->GetOutControlAnchor(), inControlAnchor),
                        OP_LOGE("PadFusionPass", "Remove OutputControl Edge failed."), return FAILED);

      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(fusion_node->GetOutControlAnchor(), inControlAnchor),
                        OP_LOGE("PadFusionPass", "Add OutputControl Edge failed."), return FAILED);
    }
  }

  // remove org_node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(pad_node), OP_LOGE("PadFusionPass", "Remove OrgNode failed."),
                    return FAILED);

  // push new_node
  fusionNodes.push_back(fusion_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "PadFusionPass SUCCESS!");

  return SUCCESS;
}

REGISTER_PASS("PadFusionPass", BUILT_IN_GRAPH_PASS, PadFusionPass);
}  // namespace fe
