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
 * \file bnupdate_reluv2_bnreduce_fusion_pass.cpp
 * \brief convert bnupdate+reluv2+conv2d+bnreduce to fusedbn2reluconvbn1
 */
#include "bnupdate_reluv2_bnreduce_fusion_pass.h"
#include <vector>
#include <string>
#include <algorithm>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"
#include "common/util/platform_info.h"

namespace fe {
static const char kBNupdateType[] = "BNTrainingUpdate";
static const char kReluV2Type[] = "ReluV2";
static const char kConv2DType[] = "Conv2D";
static const char kBNreduceType[] = "BNTrainingReduce";
static const char kDestType[] = "FusedBN2ReluConvBN1";
static const char kBNupdateId[] = "bnupdate";
static const char kReluV2Id[] = "reluv2";
static const char kConv2DId[] = "conv2d";
static const char kBNreduceId[] = "bnreduce";
static const char kAttrPads[] = "pads";
static const char kAttrStrides[] = "strides";
static const char kAttrDilations[] = "dilations";
static const char kAttrGroups[] = "groups";
static const char kAttrFactor[] = "factor";
static const char kAttrEpsilon[] = "epsilon";
static const char kCommInput[] = "x";
static const char kCommoutput[] = "y";
static const uint32_t kSupportAicoreNum = 32;
static const int kBiasIndex = 2;
static const size_t kInputAndWeightTensor = 2;

/*!
  * @brief Define bnupdate+reluv2+conv2d+bnreduce pattern.
  *
  *             bnupdate
  *                |
  *              reluv2
  *                |
  *              conv2d
  *                |
  *             bnreduce
  *                |
  *
  * @return vector<FusionPattern*> All valid patterns.
  */
vector<FusionPattern*> BNupdateReluV2Conv2DBNreducePass::DefinePatterns()
{
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = new (std::nothrow) FusionPattern("BNupdateReluV2Conv2DBNreducePass");
    FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), "new a pattern object failed."),
                      return patterns);
    pattern->AddOpDesc(kBNupdateId, {kBNupdateType})
        .AddOpDesc(kReluV2Id, {kReluV2Type})
        .AddOpDesc(kConv2DId, {kConv2DType})
        .AddOpDesc(kBNreduceId, {kBNreduceType})
        .SetInputs(kReluV2Id, {kBNupdateId})
        .SetInputs(kConv2DId, {kReluV2Id})
        .SetInputs(kBNreduceId, {kConv2DId})
        .SetOutput(kBNreduceId);
    patterns.push_back(pattern);

    return patterns;
}

/*!
  * @brief .
  *               \/
  *             bnupdate
  *               /\
  *         reluv2                       \/
  *           /\           -->   fusedbn2reluconvbn1
  *     conv2d                           /\
  *        /\
  *          bnreduce
  *             /\
  *
  * @param graph The whole graph.
  * @param mapping Matched nodes of defined pattern.
  * @param new_nodes New nodes added to graph.
  * @return Status Graph processing result.
  */
Status BNupdateReluV2Conv2DBNreducePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                std::vector<ge::NodePtr>& new_nodes)
{
    OP_LOGD(fused_op_type_.c_str(), "Enter BNupdateReluV2Conv2DBNreducePass.");
    // check the platform
    PlatformInfo platform_info;
    OptionalInfo opti_compilation_info;
    FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                      platform_info, opti_compilation_info) != SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "Get platform_info failed."),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(platform_info.soc_info.ai_core_cnt != kSupportAicoreNum,
        OP_LOGW(fused_op_type_.c_str(), "this platform not support BNupdateReluV2Conv2DBNreduce fusion."),
        return NOT_CHANGED);
    std::vector<std::string> id_list = {kBNupdateId, kReluV2Id, kConv2DId, kBNreduceId};
    std::vector<ge::NodePtr> node_list;
    std::string error_str;
    std::string fused_name;
    // entrace conditon check
    for (auto node_id : id_list) {
        ge::NodePtr node_ptr = GetNodeFromMapping(node_id, mapping);
        error_str = node_id + " node is null, fusion failed.";
        FUSION_PASS_CHECK(node_ptr == nullptr, CommonRuntimeErrLog(fused_op_type_.c_str(), error_str.c_str()),
                          return PARAM_INVALID);
        fused_name += node_ptr->GetName() + "/";
        node_list.push_back(node_ptr);
    }
    FUSION_PASS_CHECK(!AnalyzeLayers(node_list),
                      OP_LOGD(fused_op_type_.c_str(), "nothing changed on the graph."), return NOT_CHANGED);
    // create fused node
    fused_name += "nodes_fused";
    ge::OpDescPtr fused_desc(new ge::OpDesc(fused_name, kDestType));
    FUSION_PASS_CHECK(fused_desc == nullptr,
                      OP_LOGD(fused_op_type_.c_str(), "create new desc failed."), return NOT_CHANGED);
    FUSION_PASS_CHECK(!AddFusedDesc(node_list, fused_desc),
                      OP_LOGD(fused_op_type_.c_str(), "add new desc failed."), return NOT_CHANGED);
    // add node to graph
    ge::NodePtr fused_node = graph.AddNode(fused_desc);
    FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGD(fused_op_type_.c_str(), "add new fused node failed."),
                      return NOT_CHANGED);
    // link node to graph
    FUSION_PASS_CHECK(!LinkNewNode(node_list, fused_node),
                      CommonRuntimeErrLog(fused_op_type_.c_str(), "link fused node failed."), return FAILED);
    // remove original nodes
    for (auto node : node_list) {
        error_str = "remove " + node->GetName() + " node failed.";
        FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                          CommonRuntimeErrLog(fused_op_type_.c_str(), error_str.c_str()), return FAILED);
    }
    // GraphUtils::DumpGEGraphToOnnx(graph, "TestPass");
    OP_LOGD(fused_op_type_.c_str(), "Leave BNupdateReluV2Conv2DBNreducePass.");

    return SUCCESS;
}

/*!
  * @brief Check bnupdate node output single referred
  */
bool BNupdateReluV2Conv2DBNreducePass::CheckBnupdateNodeOutSingleReferred(
    const std::vector<ge::NodePtr>& node_list,
    const int idx,
    const std::vector<size_t>& ir_inputs,
    const std::vector<size_t>& expect_ref)
{
    for (size_t i = 0; i < node_list.size(); ++i) {
        auto node_desc = node_list[i]->GetOpDesc();
        FUSION_PASS_CHECK(node_desc == nullptr, OP_LOGD(fused_op_type_.c_str(), "get op desc failed."),
                          return false);
        FUSION_PASS_CHECK(ir_inputs[i] != node_desc->GetAllInputNames().size(),
                          OP_LOGD(fused_op_type_.c_str(), "the node %s inputs number is not %zu.",
                                  node_list[i]->GetName().c_str(), ir_inputs[i]),
                          return false);
        auto out_anchor = node_list[i]->GetOutDataAnchor(idx);
        FUSION_PASS_CHECK(out_anchor == nullptr,
                          OP_LOGD(fused_op_type_.c_str(), "get %s output data anchor %d failed.",
                                  node_list[i]->GetName().c_str(), idx),
                          return false);
        size_t real_num = out_anchor->GetPeerInDataNodesSize();
        if (real_num > 0 && expect_ref[i] > 1) {
            continue;
        }
        FUSION_PASS_CHECK(real_num != expect_ref[i],
                          OP_LOGD(fused_op_type_.c_str(), "expect %s output %zu referred, not %zu.",
                                  node_list[i]->GetName().c_str(), expect_ref[i], real_num),
                          return false);
    }
    return true;
}

/*!
  * @brief Check the output count and shape params.
  *
  *  x sum square_sum scale offset mean variance
  *   \   \          \  |  /      /    /
  *                               -----
  *                  bnupdate      /->    share the same
  *               /->                     memory address
  *            ---------
  *     /     /        /   |      \
  *    y  mean variance batch_mean batch_variance
  *     \
  *      reluv2
  *        \    \
  *         |     mask
  *           |        weight bias(if exist)
  *            \      /      /
  *               conv2d
  *                /  \
  *        bureduce    others
  *               |    |
  * @param node_list The bnupdate, reluv2, conv2d, bureduce node ptr list.
  * @return bool Whether the inputs and outputs are expected.
  */
bool BNupdateReluV2Conv2DBNreducePass::AnalyzeLayers(const std::vector<ge::NodePtr> &node_list)
{
    // check bnupdate node output single referred
    int idx = 0;
    std::vector<size_t> ir_inputs = {7, 1, 4, 1};
    std::vector<size_t> expect_ref = {1, 2, 2, 1};
    CheckBnupdateNodeOutSingleReferred(node_list, idx, ir_inputs, expect_ref);
    // gather tensor info
    auto conv_desc = node_list[2]->GetOpDesc();
    std::vector<int64_t> params;
    for (size_t i = 0; i < kInputAndWeightTensor; ++i) {
        auto in_tensor = conv_desc->MutableInputDesc(i);
        if (in_tensor == nullptr) {
            continue;
        }
        std::vector<int64_t> tensor_shape = in_tensor->GetOriginShape().GetDims();
        for (auto dim : tensor_shape) {
            params.push_back(dim);
        }
    }
    // not support bias now
    FUSION_PASS_CHECK(conv_desc->GetInputsSize() > kBiasIndex,
                      OP_LOGD(fused_op_type_.c_str(), "not support conv2d bias now"),
                      return false);
    // gather attribute info
    std::vector<std::string> attr_list = {kAttrStrides, kAttrPads, kAttrDilations};
    for (auto attr : attr_list) {
        std::vector<int64_t> value_list;
        ge::AttrUtils::GetListInt(conv_desc, attr, value_list);
        for (auto dim : value_list) {
            params.push_back(dim);
        }
    }
    // shape input, shape filter, shape bias, strides, pads, dilations
    const std::vector<std::vector<int64_t>> white_list = {
        {256, 56, 56, 128, 3, 3, 128, 128, 1, 2, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1},
        {256, 56, 56, 64, 3, 3, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
        };
    bool exist = false;
    for (auto item : white_list) {
        if (item == params) {
            exist = true;
        }
    }
    std::string to_print;
    for (auto i : params) {
        to_print += std::to_string(i) + ", ";
    }
    if (!exist) {
        OP_LOGD(fused_op_type_.c_str(), "params [%s] not in white list.", to_print.c_str());
        return false;
    } else {
        OP_LOGD(fused_op_type_.c_str(), "params [%s] in white list.", to_print.c_str());
    }
    return true;
}

/*!
  * @brief Replace fused desc names.
  */
bool BNupdateReluV2Conv2DBNreducePass::ReplaceFusedDescName(
    const std::vector<std::string>& ori_names,
    std::map<std::string, std::vector<std::string>>& ori_to_new,
    const string& name_attr,
    std::map<std::string, uint32_t>& new_name)
{
    for (size_t i = 0; i < ori_names.size(); ++i) {
        std::string name = ori_names[i];
        auto it = ori_to_new.find(name);
        if (it != ori_to_new.end()) {
            FUSION_PASS_CHECK((it->second).empty(), OP_LOGD(fused_op_type_.c_str(),
                "replace %s name [%s] failed.", name_attr.c_str(), name.c_str()), return false);
            name = (it->second)[0];
            (it->second).erase((it->second).begin());
        }
        new_name.insert(std::pair<std::string, uint32_t>(name, static_cast<uint32_t>(i)));
    }
    return true;
}

/*!
  * @brief Simply update fused desc in/output name and attributes.
  */
bool BNupdateReluV2Conv2DBNreducePass::UpdateDesc(const std::vector<ge::NodePtr> &node_list, ge::OpDescPtr fused_desc,
                                                  std::vector<std::string> fused_inputs,
                                                  std::vector<std::string> fused_outputs)
{
    // >>> start: replace input and output name
    std::map<std::string, std::vector<std::string>> ori_to_new = {
        {"mean", {"moving_mean"}}, {"variance", {"moving_variance"}}};
    std::map<std::string, uint32_t> new_name;
    ReplaceFusedDescName(fused_inputs, ori_to_new, "input", new_name);
    fused_desc->UpdateInputName(new_name);
    new_name.clear();
    ori_to_new = {
        {"y", {"reluv2_out", "conv2d_out"}}, {"sum", {"sum_out"}}, {"square_sum", {"square_sum_out"}},
        {"mean", {"moving_mean"}}, {"variance", {"moving_variance"}},
        {"batch_mean", {"mean"}}, {"batch_variance", {"variance"}}};
    ReplaceFusedDescName(fused_outputs, ori_to_new, "output", new_name);
    fused_desc->UpdateOutputName(new_name);
    // <<< end: replace input and output name
    // >>> start: add op attribute
    std::vector<std::string> attr_list = {kAttrFactor, kAttrEpsilon};
    for (auto attr : attr_list) {
        float value = 0.0;
        bool ret = ge::AttrUtils::GetFloat(node_list[0]->GetOpDesc(), attr, value);
        FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "get attribute [%s] failed.", attr.c_str()),
                          return false);
        ret = ge::AttrUtils::SetFloat(fused_desc, attr, value);
        FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "set attribute [%s] failed.", attr.c_str()),
                          return false);
    }
    attr_list = {kAttrStrides, kAttrPads, kAttrDilations};
    for (auto attr : attr_list) {
        std::vector<int32_t> value;
        bool ret = ge::AttrUtils::GetListInt(node_list[2]->GetOpDesc(), attr, value);
        FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "get attribute [%s] failed.", attr.c_str()),
                          return false);
        ret = ge::AttrUtils::SetListInt(fused_desc, attr, value);
        FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "set attribute [%s] failed.", attr.c_str()),
                          return false);
    }
    ge::AttrUtils::SetInt(fused_desc, kAttrGroups, 1);
    // <<< end: add op attribute
    return true;
}

/*!
  * @brief Create new concat nodes, update it's in/output name and tensor desc.
  *
  *     x sum square_sum scale offset moving_mean moving_variance  filter bias(if exist)
  *      \   \          \  |  /      /           /                /      /
  *                FusedBN2ReluConvBN1
  *             /         |     \    \        \          \    \          \       \
  *  moving_mean moving_variance mean variance reluv2_out mask conv2d_out sum_out square_sum_out
  *
  * @param node_list The bnupdate, reluv2, conv2d, bureduce node ptr list.
  * @return bool Whether new desc is added successfully.
  */
bool BNupdateReluV2Conv2DBNreducePass::AddFusedDesc(const std::vector<ge::NodePtr> &node_list, ge::OpDescPtr fused_desc)
{
    std::vector<std::string> fused_inputs;
    std::vector<std::string> fused_outputs;
    for (auto node : node_list) {
        auto node_desc = node->GetOpDesc();
        // >>> start: process orignal input name
        std::map<std::string, uint32_t> ori_inputs = node_desc->GetAllInputName();
        std::vector<std::string> input_name;
        input_name.assign(ori_inputs.size(), "");
        for (auto name_idx : ori_inputs) {
            input_name[name_idx.second] = name_idx.first;
        }
        if (node->GetType() != kBNupdateType) {
            input_name.erase(input_name.begin()); // remove x input
        }
        if (node->GetType() == kConv2DType) {
            input_name.pop_back(); // remove offset_w input
        }
        for (auto name : input_name) {
            for (auto exist : fused_inputs) {
                FUSION_PASS_CHECK((name != kCommInput && exist == name), OP_LOGD(fused_op_type_.c_str(),
                    "can't support input same name [%s].", exist.c_str()), return false);
            }
            FUSION_PASS_CHECK(fused_desc->AddInputDesc(node_desc->GetInputDesc(name)) != ge::GRAPH_SUCCESS, OP_LOGD(
                fused_op_type_.c_str(), "add input desc %s failed.", name.c_str()), return false);
        }
        fused_inputs.insert(fused_inputs.end(), input_name.begin(), input_name.end());
        // <<< end: process orignal input name
        // >>> start: process orignal output name
        std::map<std::string, uint32_t> ori_outputs = node_desc->GetAllOutputName();
        std::vector<std::string> output_name;
        output_name.assign(ori_outputs.size(), "");
        for (auto name_idx : ori_outputs) {
            output_name[name_idx.second] = name_idx.first;
        }
        if (node->GetType() == kBNupdateType) {
            output_name.erase(output_name.begin()); // remove y output
        }
        for (auto name : output_name) {
            for (auto exist : fused_outputs) {
                FUSION_PASS_CHECK((name != kCommoutput && exist == name), OP_LOGD(fused_op_type_.c_str(),
                    "can't support output same name [%s].", exist.c_str()), return false);
            }
            FUSION_PASS_CHECK(fused_desc->AddOutputDesc(node_desc->GetOutputDesc(name)) != ge::GRAPH_SUCCESS,
                              OP_LOGD(fused_op_type_.c_str(), "add output desc %s failed.", name.c_str()),
                              return false);
        }
        fused_outputs.insert(fused_outputs.end(), output_name.begin(), output_name.end());
        // <<< end: process orignal output name
    }
    bool ret = UpdateDesc(node_list, fused_desc, fused_inputs, fused_outputs);
    FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "update fused desc failed."), return false);
    // >>> start: add reference attribute
    // output tensor memery address reuse input with same IR name
    ret = ge::AttrUtils::SetBool(fused_desc, ge::ATTR_NAME_REFERENCE, true);
    FUSION_PASS_CHECK(!ret, OP_LOGD(fused_op_type_.c_str(), "add reference attribute failed."), return false);
    // <<< end: add reference attribute
    return true;
}

/*!
  * @brief Link valid input data anchor
  */
bool BNupdateReluV2Conv2DBNreducePass::LinkValidInputDataAnchor(
    ge::Node::Vistor<ge::InDataAnchorPtr>& input_anchors,
    const std::map<std::string, size_t>& node_name,
    const ge::NodePtr& fused_node,
    uint32_t& inputIdx)
{
    ge::graphStatus ret;
    for (const auto &anchor : input_anchors) {
        if (anchor == nullptr) {
            continue;
        }
        auto peer_output = anchor->GetPeerOutAnchor();
        if (peer_output == nullptr) {
            continue;
        }
        auto src_node = peer_output->GetOwnerNode();
        if (src_node == nullptr) {
            continue;
        }
        // skip inner node connection
        auto it = node_name.find(src_node->GetName());
        if (it != node_name.end()) {
            continue;
        }
        ret = peer_output->Unlink(anchor);
        FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGD(fused_op_type_.c_str(), "unlink input anchor failed."),
            return false);
        auto new_dst = fused_node->GetInDataAnchor(inputIdx);
        FUSION_PASS_CHECK(new_dst == nullptr,
            OP_LOGD(fused_op_type_.c_str(), "get fused node input %u anchor failed.", inputIdx),
            return false);
        ret = peer_output->LinkTo(new_dst);
        FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
            OP_LOGD(fused_op_type_.c_str(), "link fused node input %u anchor failed.", inputIdx),
            return false);
        inputIdx++;
    }
    return true;
}

/*!
  * @brief Link valid output data anchor
  */
bool BNupdateReluV2Conv2DBNreducePass::LinkValidOutputDataAnchor(
    ge::Node::Vistor<ge::OutDataAnchorPtr>& output_anchors,
    const std::map<std::string, size_t>& node_name,
    const ge::NodePtr& fused_node,
    uint32_t& outputIdx)
{
    ge::graphStatus ret;
    for (const auto &anchor : output_anchors) {
        if (anchor == nullptr) {
            continue;
        }
        bool has_external = false;
        auto peer_inputs = anchor->GetPeerInDataAnchors();
        for (const auto &peer_input : peer_inputs) {
            if (peer_input == nullptr) {
                continue;
            }
            auto dst_node = peer_input->GetOwnerNode();
            if (dst_node == nullptr) {
                continue;
            }
            // skip inner node connection
            auto it = node_name.find(dst_node->GetName());
            if (it != node_name.end()) {
                continue;
            }
            has_external = true;
            ret = anchor->Unlink(peer_input);
            FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
                OP_LOGD(fused_op_type_.c_str(), "unlink output anchor failed."),
                return false);
            auto new_src = fused_node->GetOutDataAnchor(outputIdx);
            FUSION_PASS_CHECK(new_src == nullptr,
                OP_LOGD(fused_op_type_.c_str(), "get fused node output %u anchor failed.", outputIdx),
                return false);
            ret = new_src->LinkTo(peer_input);
            FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
                OP_LOGD(fused_op_type_.c_str(), "link fused node output %u anchor failed.", outputIdx),
                return false);
        }
        // only external connection need to link to output
        if (has_external) {
            outputIdx++;
        }
    }
    return true;
}

/*!
  * @brief Simply link fused node to graph.
  */
bool BNupdateReluV2Conv2DBNreducePass::LinkNewNode(const std::vector<ge::NodePtr> &node_list, ge::NodePtr fused_node)
{
    uint32_t inputIdx = 0;
    uint32_t outputIdx = 2; // output valid anchor start from 2
    std::map<std::string, size_t> node_name; // map for simply identify inner nodes
    for (size_t i = 0; i < node_list.size(); ++i) {
        node_name.insert(std::pair<std::string, size_t>(node_list[i]->GetName(), i));
    }
    for (auto node : node_list) {
        // >>> start: connect input
        auto input_anchors = node->GetAllInDataAnchors();
        // get vaild input data anchor
        LinkValidInputDataAnchor(input_anchors, node_name, fused_node, inputIdx);
        // <<< end: connect input
        // >>> start: connect output
        auto output_anchors = node->GetAllOutDataAnchors();
        // get vaild output data anchor
        LinkValidOutputDataAnchor(output_anchors, node_name, fused_node, outputIdx);
        // <<< end: connect output
    }
    return true;
}

REGISTER_PASS("ZBNupdateReluV2Conv2DBNreducePass", BUILT_IN_GRAPH_PASS, BNupdateReluV2Conv2DBNreducePass);
}  // namespace fe