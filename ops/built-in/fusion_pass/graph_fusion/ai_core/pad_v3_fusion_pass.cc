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
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_PAD = "PadV3";
static const char *PAD = "PadV3";

bool PadV3FusionPass::GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                                    std::vector<int64_t> &const_data)
{
    size_t size = 0;
    uint8_t *const_data_ptr = (uint8_t *)const_tensor.GetData();
    if (const_data_ptr == nullptr) {
        OP_LOGE(op.GetName().c_str(), "const_data_ptr is null");
        return false;
    }
    if (dtype == ge::DT_INT32) {
        size = const_tensor.GetSize() / sizeof(int32_t);
        for (size_t i = 0; i < size; ++i) {
          const_data.push_back((int32_t)((*((int32_t *)const_data_ptr + i))));
          OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d",
                  (int32_t)(*((int32_t *)const_data_ptr + i)));
        }
    } else if (dtype == ge::DT_INT64) {
        size = const_tensor.GetSize() / sizeof(int64_t);
        for (size_t i = 0; i < size; ++i) {
          const_data.push_back(((int64_t)(*((int64_t *)const_data_ptr + i))));
          OP_LOGD(op.GetName().c_str(), "const data int64 fusion pass ====== %d",
                  (int64_t)(*((int64_t *)const_data_ptr + i)));
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "not support this type");
        return false;
    }
    return true;
}

vector<FusionPattern *> PadV3FusionPass::DefinePatterns()
{
    vector < FusionPattern * > patterns;

    // pad fusion to pad_d
    FusionPattern *pattern = new(std::nothrow) FusionPattern("PadV3Fusion");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
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
                              OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode1->GetName().c_str()),
                              return false);
            OP_LOGD(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode1->GetName().c_str());
        } else {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:[%s] have output link to other node.", constNode1->GetName().c_str());
        }
    }

    if (!ge::OpDescUtils::ClearInputDesc(pad_desc, index)) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to clear input desc[%d]", index);
    }

    return true;
}

Status PadV3FusionPass::PadMoveConsttoAttr(ge::ComputeGraph &graph, ge::NodePtr &pad_node)
{
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
    Tensor const_tensor;
    if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", const_tensor)) {
        OP_LOGE(op.GetName().c_str(), "Get GetInputConstData failed ");
        return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("paddings").GetDataType();

    std::vector<int64_t> pad_value;
    if (!GetConstValue(op, const_tensor, dtype, pad_value)) {
        OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
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
    FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "pad_v3_node's OpDesc is null, fusion failed."), return PARAM_INVALID);
    ge::AttrUtils::SetListListInt(pad_desc, "paddings", paddings);

    // translate constant_values to attr
    if (pad_desc->MutableInputDesc(pad_desc->GetInputIndexByName("constant_values")) != nullptr) {
        if (ge::GRAPH_SUCCESS != op.GetInputConstData("constant_values", const_tensor)) {
            OP_LOGE(op.GetName().c_str(), "Get GetInputConstData failed ");
            return GRAPH_FAILED;
        }
        dtype = op.GetInputDesc("constant_values").GetDataType();

        vector<int64_t> const_value;
        if (!GetConstValue(op, const_tensor, dtype, const_value)) {
            OP_LOGE(op.GetName().c_str(), "Get Const Value failed ");
            return GRAPH_FAILED;
        }
        ge::AttrUtils::SetInt(pad_desc, "constant_values", const_value.at(0));
    }

    // remove input node as index descend
    FUSION_PASS_CHECK(!AutoRemoveInput(graph, pad_node, op, "constant_values"), OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "remove input constant_values failed, fusion failed."), return GRAPH_FAILED);
    FUSION_PASS_CHECK(!AutoRemoveInput(graph, pad_node, op, "paddings"), OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "remove input paddings failed, fusion failed."), return GRAPH_FAILED);

    return SUCCESS;
}

Status PadV3FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes)
{
    // get pad node and node-desc
    ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PAD, mapping);
    FUSION_PASS_CHECK(pad_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "pad_v3_node is null, fusion failed."),
                      return PARAM_INVALID);

    ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
    FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                      "pad_v3_node's OpDesc is null, fusion failed."), return PARAM_INVALID);

    vector<int64_t> dims = pad_desc->GetOutputDesc("y").GetShape().GetDims();
    for (int64_t ele : dims) {
        if (ele == UNKNOWN_DIM) {
            OP_LOGI(FUSED_OP_TYPE.c_str(), "It is unknown shape, not changed");
            return NOT_CHANGED;
        }
    }

    if (PadMoveConsttoAttr(graph, pad_node) != SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), " PadMoveConsttoAttr failed.");
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
