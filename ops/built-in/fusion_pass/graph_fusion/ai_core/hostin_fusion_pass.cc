/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief instance norm fusion pass(instance norm --> pure instance norm)
 *
 */

#include "hostin_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
    static const string PATTERN_INInference = "INInferV2";
    static const string ININFERENCE = "INInferV2";
    static const string EPSILON = "epsilon";

    vector<FusionPattern*> HostINFusionPass::DefinePatterns()
    {
        vector<FusionPattern*> patterns;
        FusionPattern* pattern =
            new(std::nothrow) FusionPattern("HostINFusionPass");
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter HostINFusionPass::DefinePatterns.");
        FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
                 return patterns);

        pattern->AddOpDesc(PATTERN_INInference, {ININFERENCE})
            .SetOutput(PATTERN_INInference);
        patterns.push_back(pattern);

        return patterns;
    }

    Status HostINFusionPass::Fusion(
        ge::ComputeGraph& graph, Mapping& mapping,
        vector<ge::NodePtr>& newNodes)
    {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GoINhost");
        ge::NodePtr inNode = GetNodeFromMapping(PATTERN_INInference, mapping);
        FUSION_PASS_CHECK(inNode == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Node Ininfenced is null, fusion failed."),
                 return PARAM_INVALID);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "check INhost");
        FUSION_PASS_CHECK(CheckParameter(inNode) != SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Check INInferenceD param failed."), return PARAM_INVALID);

        OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion INhost");
        return INFuison(graph, inNode, newNodes);
    }

    Status HostINFusionPass::CheckParameter(ge::NodePtr& inNodePtr)
    {
        // get psroipooling node inputs.
        Node::Vistor<NodePtr> inNodes = inNodePtr->GetInDataNodes();
        FUSION_PASS_CHECK((inNodes.size() != 5),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "INInference input nodes num(%d) != 5",
                         inNodes.size()), return PARAM_INVALID);
        return SUCCESS;
    }

    Status HostINFusionPass::SetAttrValueForNewNode(
        const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr)
    {
        // get and update output_dim
        ge::GeAttrValue epsValue;
        FUSION_PASS_CHECK(preOpDescPtr->GetAttr(EPSILON, epsValue) ==
                 ge::GRAPH_FAILED,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Get attr %s from node %s error",
                         EPSILON.c_str(), preOpDescPtr->GetName().c_str()),
                 return PARAM_INVALID);

        FUSION_PASS_CHECK(newOpDescPtr->SetAttr(EPSILON, epsValue) ==
                 ge::GRAPH_FAILED,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr %s to node %s error", EPSILON.c_str(),
                         newOpDescPtr->GetName().c_str()),
                 return PARAM_INVALID);

        return SUCCESS;
    }

    Status HostINFusionPass::INFuison(
        ge::ComputeGraph& graph,
        ge::NodePtr& inNodePtr, vector<ge::NodePtr>& newNodes)
    {
        // check conv op desc is null or not

        ge::OpDescPtr inOpDescPtr = inNodePtr->GetOpDesc();
        FUSION_PASS_CHECK(inOpDescPtr == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                         inOpDescPtr->GetName().c_str()), return PARAM_INVALID);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "NODE %s 1", inOpDescPtr->GetName().c_str());

        // create bnhost opdesc
        std::vector<string> inputType;
        Node::Vistor<NodePtr> inNodes = inNodePtr->GetInDataNodes();

        for (const auto& node : inNodes) {
            inputType.push_back(node->GetType());
            OP_LOGI(FUSED_OP_TYPE.c_str(), "dtype %s 1", node->GetType().c_str());
        }
        vector<string>::iterator temp;
        temp = find(inputType.begin(), inputType.end(), "Const");
        if (temp == inputType.end()) {
            return NOT_CHANGED;
        } else {
            // create transpose opdesc
            std::shared_ptr<ge::OpDesc> inhostOpDescPtr = nullptr;
            inhostOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_inhost", "InHost");

            std::shared_ptr<ge::OpDesc> ininferOpDescPtr = nullptr;
            ininferOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_v2D", "INInferV2D");

            FUSION_PASS_CHECK(SetAttrValueForNewNode(inOpDescPtr, inhostOpDescPtr) != SUCCESS,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Update output_dim and group_size failed."), return FAILED);

            // get Inhost input
            ge::GeTensorDesc varInputTensorDesc = inOpDescPtr->GetInputDesc(4);
            // get inhost output
            ge::GeTensorDesc varSqrtOutputTensorDesc = inOpDescPtr->GetInputDesc(4);

            //get INInferenced input
            ge::GeTensorDesc ininferdataInputTensorDesc = inOpDescPtr->GetInputDesc(0);
            ge::GeTensorDesc ininfergammaInputTensorDesc = inOpDescPtr->GetInputDesc(1);
            ge::GeTensorDesc ininferbetaInputTensorDesc = inOpDescPtr->GetInputDesc(2);
            ge::GeTensorDesc ininfermeanInputTensorDesc = inOpDescPtr->GetInputDesc(3);
            ge::GeTensorDesc ininfervarInputTensorDesc = inOpDescPtr->GetInputDesc(4);


            //get INInferenced output

            ge::GeTensorDesc ininferOutputTensorDesc = inOpDescPtr->GetOutputDesc(0);
            ge::GeTensorDesc ininferBmeanOutputTensorDesc = inOpDescPtr->GetOutputDesc(1);
            ge::GeTensorDesc ininferBvarianceOutputTensorDesc = inOpDescPtr->GetOutputDesc(2);


            // update output origin shape of pad;
            inhostOpDescPtr->AddInputDesc("variance", varInputTensorDesc);
            inhostOpDescPtr->AddOutputDesc("variance_sqrt", varSqrtOutputTensorDesc);

            ininferOpDescPtr->AddInputDesc("x", ininferdataInputTensorDesc);
            ininferOpDescPtr->AddInputDesc("gamma", ininfergammaInputTensorDesc);
            ininferOpDescPtr->AddInputDesc("beta", ininferbetaInputTensorDesc);
            ininferOpDescPtr->AddInputDesc("mean", ininfermeanInputTensorDesc);
            ininferOpDescPtr->AddInputDesc("variance", ininfervarInputTensorDesc);

            ininferOpDescPtr->AddOutputDesc("y", ininferOutputTensorDesc);
            ininferOpDescPtr->AddOutputDesc("batch_mean", ininferBmeanOutputTensorDesc);
            ininferOpDescPtr->AddOutputDesc("batch_variance", ininferBvarianceOutputTensorDesc);

            // output of inhost op
            ininferOpDescPtr->AddInputDesc("variance_sqrt", varSqrtOutputTensorDesc);


            // add SwapCo node to graph
            ge::NodePtr inhostNodePtr = graph.AddNode(inhostOpDescPtr);
            ge::NodePtr ininferNodePtr = graph.AddNode(ininferOpDescPtr);
            newNodes.push_back(inhostNodePtr);
            newNodes.push_back(ininferNodePtr);

            FUSION_PASS_CHECK(inhostNodePtr == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: inhostNodePtr is null, fusion failed."),
                     return FAILED);
            FUSION_PASS_CHECK(ininferNodePtr == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: ininferNodePtr is null, fusion failed."),
                     return FAILED);


            // x gamma, beta, mean, variance,
            // y, batch_mean, batch_variance, ins_training, momentum, epsilon

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
                ininferNodePtr->GetInDataAnchor(0)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(0)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             ininferNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(1)->GetPeerOutAnchor(),
                ininferNodePtr->GetInDataAnchor(1)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(1)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             ininferNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(2)->GetPeerOutAnchor(),
                ininferNodePtr->GetInDataAnchor(2)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(2)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             ininferNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(3)->GetPeerOutAnchor(),
                ininferNodePtr->GetInDataAnchor(3)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(3)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             ininferNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(4)->GetPeerOutAnchor(),
                ininferNodePtr->GetInDataAnchor(4)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(4)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             ininferNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(4)->GetPeerOutAnchor(),
                inhostNodePtr->GetInDataAnchor(0)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(4)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             inhostNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inhostNodePtr->GetOutAnchor(0),
                ininferNodePtr->GetInDataAnchor(5)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inhostNodePtr->GetName().c_str(),
                             inNodePtr->GetName().c_str()),
                     return FAILED);



            // add the output of INInferenced edge
            // copy output edge
            size_t outanchorsize = inNodePtr->GetAllOutAnchors().size();
            for (size_t outindex = 0; outindex < outanchorsize; outindex++) {
                for (auto inDataAnchor : inNodePtr->GetOutDataAnchor(outindex)->GetPeerInDataAnchors()) {
                    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(inNodePtr->GetOutDataAnchor(outindex),
                                                        inDataAnchor) != SUCCESS,
                             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove inhost out data edge failed."), return FAILED);
                    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(ininferNodePtr->GetOutDataAnchor(outindex),
                                                     inDataAnchor) != SUCCESS,
                             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add inhost out data edge failed."), return FAILED);
                }
            }

            // remove Normalize from graph
            FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(inNodePtr),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "remove inNodePtr node[%s] failed", inNodePtr->GetName().c_str()),
                     return FAILED);
            return SUCCESS;
        }
    }

    REGISTER_PASS("HostINFusionPass", BUILT_IN_GRAPH_PASS, HostINFusionPass);
}