/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(pad --> pad_d)
 *
 */

#include "pad_fusion_pass.h"

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
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_PAD = "Pad";
static const char *PAD = "Pad";

bool PadFusionPass::GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                          std::vector<int64_t> &const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *) const_tensor.GetData();
    if(const_data_ptr == nullptr){
      OP_LOGE(op.GetName().c_str(), "const_data_ptr is null");
    }
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *) const_tensor.GetData();
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

vector<FusionPattern *> PadFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  // pad fusion to pad_d
  FusionPattern *pattern = new(std::nothrow) FusionPattern("PadFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
  return patterns);

  pattern->AddOpDesc(PATTERN_PAD, {PAD})
      .SetOutput(PATTERN_PAD);

  patterns.push_back(pattern);

  return patterns;
}

Status PadFusionPass::PadMoveConsttoAttr(ge::ComputeGraph &graph, ge::NodePtr &pad_node, const string &attr_name, int32_t index) {

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
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

Status PadFusionPass::Fusion(ge::ComputeGraph &graph,
                             Mapping &mapping,
                             vector<ge::NodePtr> &fusionNodes)
{
  // get pad node and node-desc
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PAD, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_node is null, fusion failed."),
           return PARAM_INVALID);

  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_node's OpDesc is null, fusion failed."),
           return PARAM_INVALID);
  vector<int64_t> dims = pad_desc->GetOutputDesc("y").GetShape().GetDims();
  for(int64_t ele : dims){
    if (ele == UNKNOWN_DIM) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "It is unknown shape, not changed");
    return NOT_CHANGED;
    }
  }

  std::vector<PassAttrInfo> attr_infos = {
        {1, "paddings", "SetInt"}
    };
    const std::string fusion_op_type = "PadD";
    ge::OpDescPtr fusionDescPtr =
        PatternFusionUtil::GetFusionOpDesc(pad_node, fusion_op_type, attr_infos);
    FUSION_PASS_CHECK(fusionDescPtr == nullptr,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),return PARAM_INVALID);

  if (PadMoveConsttoAttr(graph, pad_node, "paddings", 1) != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), " PadMoveConsttoAttr failed.");
    return PARAM_INVALID;
  }

  vector<bool> is_input_const = {false};
  pad_desc->SetIsInputConst(is_input_const);

  // set op type Pad->PadD
  pad_desc->SetType("PadD");
  fusionNodes.push_back(pad_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "pad_node fusion SUCCESSS!");

  return SUCCESS;
}

REGISTER_PASS("PadFusionPass", BUILT_IN_GRAPH_PASS, PadFusionPass);
}
