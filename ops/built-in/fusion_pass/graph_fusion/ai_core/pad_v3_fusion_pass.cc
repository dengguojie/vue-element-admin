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

/*!
 * \file pad_v3_fusion_pass.cpp
 * \brief split fusion pass(pad_v3 --> pad_v3_d)
 */
#include "pad_v3_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

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

using namespace ge;
namespace fe {
static const std::string PATTERN_PAD = "PadV3";
static const char *PAD = "PadV3";

#define GET_CONST_DATA(DTYPE, TYPE, TENSOR, DATA)   \
  case (DTYPE): { \
    if (!GetConstDataTemplate<TYPE>(TENSOR, DATA)) { \
      VECTOR_FUSION_INNER_ERR_REPORT(PAD, "get const_data failed"); \
      return false; \
    } \
    break; \
  } \

template<typename T>
bool GetConstDataTemplate(const ge::Tensor &const_tensor, std::vector<int64_t> &const_data)
{
  size_t size = 0;
  uint8_t *const_data_ptr = (uint8_t *)const_tensor.GetData();
  if (const_data_ptr == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(PAD, "const_data_ptr is null");
    return false;
  }
  size = const_tensor.GetSize() / sizeof(T);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back(static_cast<int64_t>(*((T *)const_data_ptr + i)));
    OP_LOGD(PAD, static_cast<int64_t>(*((T *)const_data_ptr + i)));
  }
  return true;
}

bool PadV3FusionPass::GetConstValue(const ge::Tensor &const_tensor, const DataType &dtype,
                                    std::vector<int64_t> &const_data)
{
  switch (dtype) {
    GET_CONST_DATA(DT_INT8, int8_t, const_tensor, const_data)
    GET_CONST_DATA(DT_INT16, int16_t, const_tensor, const_data)
    GET_CONST_DATA(DT_INT32, int32_t, const_tensor, const_data)
    GET_CONST_DATA(DT_INT64, int64_t, const_tensor, const_data)
    GET_CONST_DATA(DT_UINT8, uint8_t, const_tensor, const_data)
    GET_CONST_DATA(DT_UINT16, uint16_t, const_tensor, const_data)
    GET_CONST_DATA(DT_UINT32, uint32_t, const_tensor, const_data)
    GET_CONST_DATA(DT_UINT64, uint64_t, const_tensor, const_data)
    GET_CONST_DATA(DT_FLOAT, float, const_tensor, const_data)
    GET_CONST_DATA(DT_FLOAT16, uint16_t, const_tensor, const_data)
    GET_CONST_DATA(DT_DOUBLE, double, const_tensor, const_data)
    GET_CONST_DATA(DT_BOOL, bool, const_tensor, const_data)
    default:
      VECTOR_FUSION_INNER_ERR_REPORT(PAD, "get const_data failed");
      return false;
  }
  return true;
}

vector<FusionPattern *> PadV3FusionPass::DefinePatterns()
{
  vector < FusionPattern * > patterns;

  // pad fusion to pad_d
  FusionPattern *pattern = new(std::nothrow) FusionPattern("PadV3Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PAD, {PAD}).SetOutput(PATTERN_PAD);

  patterns.push_back(pattern);

  return patterns;
}

bool PadV3FusionPass::AutoRemoveInput(ge::ComputeGraph &graph, ge::NodePtr &pad_node, ge::Operator &op,
                                      const string input_name)
{
  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  int index = pad_desc->GetInputIndexByName(input_name);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "input [%s] index = [%d]", input_name.c_str(), index);
  ge::InDataAnchorPtr pad_anchor_ptr1 = pad_node->GetInDataAnchor(index);
  ge::NodeUtils::ClearInDataAnchor(pad_node, pad_anchor_ptr1);

  // delete input node, edge if has
  ge::OutDataAnchorPtr const_anchor_ptr = pad_anchor_ptr1->GetPeerOutAnchor();
  if (const_anchor_ptr != nullptr) {
    ge::GraphUtils::RemoveEdge(const_anchor_ptr, pad_anchor_ptr1);
    ge::NodePtr constNode1 = const_anchor_ptr->GetOwnerNode();
    if (PatternFusionUtil::GetOutEdgeSize(constNode1) == 0) {
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode1),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode1->GetName().c_str()),
                        return false);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode1->GetName().c_str());
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.", constNode1->GetName().c_str());
    }
  }

  if (!ge::OpDescUtils::ClearInputDesc(pad_desc, index)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to clear input desc[%d]", index);
    return false;
  }

  return true;
}

Status PadV3FusionPass::PadMoveConsttoAttr(ge::ComputeGraph &graph, ge::NodePtr &pad_node)
{
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  Tensor const_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", const_tensor)) {
    VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "Get GetInputConstData failed ");
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("paddings").GetDataType();

  std::vector<int64_t> pad_value;
  if (!GetConstValue(const_tensor, dtype, pad_value)) {
    VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "Get Paddings Const Value failed ");
    return GRAPH_FAILED;
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
                    "pad_v3_node's OpDesc is null, fusion failed."), return PARAM_INVALID);
  ge::AttrUtils::SetListListInt(pad_desc, "paddings", paddings);

  // translate constant_values to attr
  if (pad_desc->MutableInputDesc(pad_desc->GetInputIndexByName("constant_values")) != nullptr) {
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("constant_values", const_tensor)) {
      VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "Get GetInputConstData failed ");
      return GRAPH_FAILED;
    }
    dtype = op.GetInputDesc("constant_values").GetDataType();

    vector<int64_t> const_value;
    if (!GetConstValue(const_tensor, dtype, const_value)) {
      VECTOR_FUSION_INNER_ERR_REPORT(op.GetName().c_str(), "Get Const Value failed ");
      return GRAPH_FAILED;
    }
    ge::AttrUtils::SetInt(pad_desc, "constant_values", const_value.at(0));
  }

  // remove input node as index descend
  FUSION_PASS_CHECK(!AutoRemoveInput(graph, pad_node, op, "constant_values"), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove input constant_values failed, fusion failed."), return GRAPH_FAILED);
  FUSION_PASS_CHECK(!AutoRemoveInput(graph, pad_node, op, "paddings"), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove input paddings failed, fusion failed."), return GRAPH_FAILED);

  return SUCCESS;
}

Status PadV3FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes)
{
  bool is_dynamic_shape = false;
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PAD, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "pad_v3_node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "pad_v3_node's OpDesc is null, fusion failed."), return PARAM_INVALID);
  // get fuzz build attr
  ge::AttrUtils::GetBool(pad_node->GetOpDesc(), ge::ATTR_NAME_FUZZ_BUILD, is_dynamic_shape);

  if (is_dynamic_shape) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "is dynamic shape.");
    return NOT_CHANGED;
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "is not dynamic shape.");
  }

  vector<int64_t> dims = pad_desc->GetOutputDesc("y").GetShape().GetDims();
  for (int64_t ele : dims) {
    if (ele == UNKNOWN_DIM) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "It is unknown shape, not changed");
      return NOT_CHANGED;
    }
  }

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  if (op.GetInputDesc("x").GetDataType() == ge::DT_INT64) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Inputs dtype is int64, not changed");
    return NOT_CHANGED;
  }

  if (PadMoveConsttoAttr(graph, pad_node) != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), " PadMoveConsttoAttr failed.");
    return PARAM_INVALID;
  }

  vector<bool> is_input_const = {false};
  pad_desc->SetIsInputConst(is_input_const);

  // set op type PadV3->PadV3D
  pad_desc->SetType("PadV3D");
  fusionNodes.push_back(pad_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "pad_v3_node fusion SUCCESSS!");

  return SUCCESS;
}

REGISTER_PASS("PadV3FusionPass", BUILT_IN_GRAPH_PASS, PadV3FusionPass);
}