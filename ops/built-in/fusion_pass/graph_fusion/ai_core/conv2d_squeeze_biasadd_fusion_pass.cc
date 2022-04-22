/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.
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
 * \file conv2d_squeeze_biasadd_fusion_pass.cpp
 * \brief conv-squeeze-biasadd fusion pass(conv2d-squeeze-biasadd --> conv2d-biasadd-squeeze)
 * The graph struct need to adapt is shown as follows:
 *
 *                     conv2d                               conv2d      bias
 *                       |                                     |       /
 *                    squeeze    bias                       biasadd
 *                       |      /           ->                 |
 *                    biasadd                               squeeze
 *                       |                                     |
 *                     output                                output
 */
#include "conv2d_squeeze_biasadd_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

using namespace ge;
namespace fe {
static const string kOpTypeConv = "Conv2D";
static const string kOpTypeSqueeze = "Squeeze";
static const string kOpTypeBiasadd = "BiasAdd";
static const string kDescConv = "conv2d";
static const string kDesSqueeze = "squeeze";
static const string kDescBiasadd = "biasadd";
static const string VARIABLE = "Variable";

vector<FusionPattern*> Conv2DSqueezeBiasaddFusionPass::DefinePatterns()
{
    vector<FusionPattern*> patterns;
    string pass_name = "Conv2DSqueezeBiasaddFusionPass";
    FusionPattern* pattern = new (std::nothrow) FusionPattern(pass_name);
    FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "new an object failed."),
                      return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
    pattern->AddOpDesc(kDescConv, {kOpTypeConv})
        .AddOpDesc(kDesSqueeze, {kOpTypeSqueeze})
        .AddOpDesc(kDescBiasadd, {kOpTypeBiasadd})
        .SetInputs(kDesSqueeze, {kDescConv})
        .SetInputs(kDescBiasadd, {kDesSqueeze})
        .SetOutput(kDescBiasadd);
    patterns.push_back(pattern);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());
    return patterns;
}

Status Conv2DSqueezeBiasaddFusionPass::Fusion(ge::ComputeGraph& graph,
                                              Mapping& mapping,
                                              vector<ge::NodePtr>& new_nodes)
{
    (void) graph;
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter Conv2DSqueezeBiasaddFusionPass");
    ge::NodePtr conv_node = GetNodeFromMapping(kDescConv, mapping);
    ge::NodePtr squeeze_node = GetNodeFromMapping(kDesSqueeze, mapping);
    ge::NodePtr biasadd_node = GetNodeFromMapping(kDescBiasadd, mapping);
    FUSION_PASS_CHECK(conv_node == nullptr,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Node conv2d is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(squeeze_node == nullptr,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Node squeeze is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(biasadd_node == nullptr,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Node biasadd is null, fusion failed."),
                      return PARAM_INVALID);
    ge::OpDescPtr out_op_conv_ptr = conv_node->GetOpDesc();
    ge::OpDescPtr in_op_biasadd_ptr = biasadd_node->GetOpDesc();
    FUSION_PASS_CHECK(out_op_conv_ptr == nullptr,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Node conv2d OpDesc is null, fusion failed."),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(in_op_biasadd_ptr == nullptr,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Node biasadd OpDesc is null, fusion failed."),
                      return PARAM_INVALID);
    vector<int64_t> biasadd_input_shape = biasadd_node->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();
    if (biasadd_input_shape.size() != 1) {
        OP_LOGW(FUSED_OP_TYPE.c_str(), "when biasadd's second inputdata's dimension is not 1 does not need changed");
        return PARAM_INVALID;
    }

    auto nodeInfrontOfAdd = biasadd_node->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
    bool case_training = (nodeInfrontOfAdd->GetType() == VARIABLE);
    if (case_training) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "We do not support Conv2DSqueezeBiasaddFusionPass in training mode.");
        return NOT_CHANGED;
    }

    // update biasadd shape and format
    ge::GeTensorDesc biasx_input_desc = in_op_biasadd_ptr->GetInputDesc(0);
    ge::GeTensorDesc biasx_output_desc = in_op_biasadd_ptr->GetOutputDesc(0);
    biasx_input_desc.SetOriginShape(out_op_conv_ptr->GetOutputDesc(0).GetShape());
    biasx_input_desc.SetShape(out_op_conv_ptr->GetOutputDesc(0).GetShape());
    biasx_output_desc.SetOriginShape(out_op_conv_ptr->GetOutputDesc(0).GetShape());
    biasx_output_desc.SetShape(out_op_conv_ptr->GetOutputDesc(0).GetShape());
    biasadd_node->GetOpDesc()->UpdateInputDesc(0, biasx_input_desc);
    biasadd_node->GetOpDesc()->UpdateOutputDesc(0, biasx_output_desc);

    // unlink and link edge
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(squeeze_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                 squeeze_node->GetInDataAnchor(0)) != SUCCESS,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove conv2d-squeeze edge failed"),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(biasadd_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                 biasadd_node->GetInDataAnchor(0)) != SUCCESS,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Remove squeeze-biasadd edge failed"),
                      return FAILED);

    for (auto inDataAnchor : biasadd_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(biasadd_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove outnode edges failed"),
                          return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(squeeze_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add outnode edge failed"),
                          return FAILED);
    }
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                                              biasadd_node->GetInDataAnchor(0)) != SUCCESS,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Add edge from conv2d to biasadd failed"),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(biasadd_node->GetOutDataAnchor(0),
                                              squeeze_node->GetInDataAnchor(0)) != SUCCESS,
                      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Add edge from biasadd to squeeze failed."),
                      return FAILED);
    return SUCCESS;
}
REGISTER_PASS("Conv2DSqueezeBiasaddFusionPass", BUILT_IN_GRAPH_PASS, Conv2DSqueezeBiasaddFusionPass);
}  // namespace fe
