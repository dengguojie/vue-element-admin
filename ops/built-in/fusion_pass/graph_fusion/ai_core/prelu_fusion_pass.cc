/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief LayerNormGrad fusion pass(LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 *
 */

#include "prelu_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace std;

namespace fe {

// node type
static const string PRELU_NODE = "PRelu";
static const string RELU_NODE = "Relu";
static const string NEG_NODE = "Neg";
static const string MUL_NODE = "Mul";
static const string ADD_NODE = "Add";

// node name id
static const string PATTERN_RELU = "Relu";
static const string PATTERN_NEG = "Neg";
static const string PATTERN_RELU1 = "Relu_1";
static const string PATTERN_NEG1 = "Neg_1";
static const string PATTERN_MUL = "mul";
static const string PATTERN_ADD = "add";

/*
before:
             Conv2D
            /      \
           /        \
          /         Neg1
         /            /
        /    Neg    Relu1
       /        \    /
        \        Mul1
        Relu     /
           \    /
            Add
             |
          MaxPool

 after:
             Conv2D
                |
                |
              PRelu
                |
             MaxPool
*/

vector<FusionPattern *> PReluFusionPass::DefinePatterns() {

    vector < FusionPattern*> patterns;

    FusionPattern *pattern = new(std::nothrow) FusionPattern("PReluFusionPass");

    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

    pattern->AddOpDesc(PATTERN_RELU, {RELU_NODE})
            .AddOpDesc(PATTERN_NEG, {NEG_NODE})
            .AddOpDesc(PATTERN_RELU1, {RELU_NODE})
            .AddOpDesc(PATTERN_NEG1, {NEG_NODE})
            .AddOpDesc(PATTERN_MUL, {MUL_NODE})
            .AddOpDesc(PATTERN_ADD, {ADD_NODE})
            .SetInputs(PATTERN_RELU1, {PATTERN_NEG1})// Neg_1->Relu_1
            .SetInputs(PATTERN_MUL, {PATTERN_NEG, PATTERN_RELU1})// Neg->mul,  Relu_1->mul
            .SetInputs(PATTERN_ADD, {PATTERN_RELU, PATTERN_MUL})// mul->add, Relu->add
            .SetOutput(PATTERN_ADD);

    patterns.push_back(pattern);

    return patterns;
}

Status PReluFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) {

    OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion PReluFusionPass!");

    ge::NodePtr addNode = GetNodeFromMapping(PATTERN_ADD, mapping);
    ge::NodePtr reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);
    ge::NodePtr negNode = GetNodeFromMapping(PATTERN_NEG, mapping);
    ge::NodePtr relu1Node = GetNodeFromMapping(PATTERN_RELU1, mapping);
    ge::NodePtr neg1Node = GetNodeFromMapping(PATTERN_NEG1, mapping);
    ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
    FUSION_PASS_CHECK(addNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "addNode is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(reluNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "reluNode is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(negNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "negNode is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(relu1Node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "relu1Node is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(neg1Node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "neg1Node is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(mulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "mulNode is null, fusion failed."), return PARAM_INVALID);

    int inputSize = 0;
    inputSize = reluNode->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "relu node size is [%lu], which not equal to 1.", inputSize), return NOT_CHANGED);

    inputSize = negNode->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "neg node size is [%lu], which not equal to 1.", inputSize), return NOT_CHANGED);

    inputSize = relu1Node->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "relu1 node size is [%lu], which not equal to 1.", inputSize), return NOT_CHANGED);

    inputSize = neg1Node->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "neg1 node size is [%lu], which not equal to 1.", inputSize), return NOT_CHANGED);

    inputSize = mulNode->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 2, OP_LOGI(FUSED_OP_TYPE.c_str(), "mul node size is [%lu], which not equal to 2.", inputSize), return NOT_CHANGED);

    inputSize = addNode->GetInDataNodes().size();
    FUSION_PASS_CHECK(inputSize != 2, OP_LOGI(FUSED_OP_TYPE.c_str(), "add node size is [%lu], which not equal to 2.", inputSize), return NOT_CHANGED);

    std::string preluNodeName = addNode->GetName() + "_" + "prelu";
    std::shared_ptr<ge::OpDesc> preluOpdesc = std::make_shared<ge::OpDesc>(preluNodeName, PRELU_NODE);
    FUSION_PASS_CHECK(preluOpdesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "preluOpdesc is null, fusion failed."), return PARAM_INVALID);

    ge::GeTensorDesc neg1InputDesc = neg1Node->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(preluOpdesc->AddInputDesc(0, neg1InputDesc) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "add neg1Node input desc failed."), return FAILED);

    ge::GeTensorDesc negInputDesc = negNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(preluOpdesc->AddInputDesc(1, negInputDesc) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "add negNode input desc failed."), return FAILED);

    ge::GeTensorDesc addOutputDesc = addNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(preluOpdesc->AddOutputDesc(addOutputDesc) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "add addNode output desc failed."), return FAILED);

    ge::NodePtr preluNode = graph.AddNode(preluOpdesc);
    preluNode->GetOpDesc()->SetType(PRELU_NODE);
    newNodes.push_back(preluNode);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
        neg1Node->GetInDataAnchor(0)->GetPeerOutAnchor(),
        preluNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between neg1Node and preluNode failed."), return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
        negNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
        preluNode->GetInDataAnchor(1)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between negNode and preluNode failed."), return FAILED);

    for (auto inDataAnchor : addNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(addNode->GetOutDataAnchor(0),
            inDataAnchor) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preluNode->GetOutDataAnchor(0),
            inDataAnchor) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
    }

    FUSION_PASS_CHECK(graph.RemoveNode(addNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove addNode failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reluNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove reluNode failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(negNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove negNode failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(relu1Node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove relu1Node failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(neg1Node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove neg1Node failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(mulNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove mulNode failed."), return FAILED);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave graph fusion PReluFusionPass!");

    return SUCCESS;
}

REGISTER_PASS("PReluFusionPass", BUILT_IN_GRAPH_PASS, PReluFusionPass);

}
