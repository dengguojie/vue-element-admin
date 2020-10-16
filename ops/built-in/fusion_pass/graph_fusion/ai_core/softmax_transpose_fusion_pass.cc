/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief instance norm fusion pass(instance norm --> pure instance norm)
 *
 */

#include "softmax_transpose_fusion_pass.h"

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
    static const string PATTERN_Softmax = "SoftmaxV2";
    static const string SOFTMAX = "SoftmaxV2";
    static const string AXIS = "axes";

    vector<FusionPattern*> softmaxTransFusionPass::DefinePatterns()
    {
        vector<FusionPattern*> patterns;
        FusionPattern* pattern =
            new(std::nothrow) FusionPattern("softmaxTransFusionPass");
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter softmaxTransFusionPass::DefinePatterns.");
        FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
                 return patterns);

        pattern->AddOpDesc(PATTERN_Softmax, {SOFTMAX})
            .SetOutput(PATTERN_Softmax);
        patterns.push_back(pattern);

        return patterns;
    }

    Status softmaxTransFusionPass::Fusion(
        ge::ComputeGraph& graph, Mapping& mapping,
        vector<ge::NodePtr>& newNodes)
    {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GoSoftmaxV2");
        ge::NodePtr inNode = GetNodeFromMapping(PATTERN_Softmax, mapping);
        FUSION_PASS_CHECK(inNode == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Node SoftmaxV2 is null, fusion failed."),
                 return PARAM_INVALID);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "check SoftmaxV2");
        FUSION_PASS_CHECK(CheckParameter(inNode) != SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Check SoftmaxV2 param failed."), return PARAM_INVALID);

        OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion SoftmaxV2");
        return INFuison(graph, inNode, newNodes);
    }

    Status softmaxTransFusionPass::CheckParameter(ge::NodePtr& inNodePtr)
    {
        // get psroipooling node inputs.
        Node::Vistor<NodePtr> inNodes = inNodePtr->GetInDataNodes();
        FUSION_PASS_CHECK((inNodes.size() != 1),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "the input data size num(%d) != 1",
                         inNodes.size()), return PARAM_INVALID);
        return SUCCESS;
    }

    Status softmaxTransFusionPass::SetAttrValueForNewNode(
        const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr, int64_t shapeLens)
    {
        vector<int32_t> axisValue;
        ge::AttrUtils::GetListInt(preOpDescPtr, AXIS, axisValue);

        //change softmax axis at axis == [-1,lens-1, -2 ,lens-2]
        for (uint32_t i = 0; i < axisValue.size(); i++) {
            if (axisValue[i] == -1 || axisValue[i] == (shapeLens - 1)) {
                axisValue[i] = shapeLens - 2;
            } else if (axisValue[i] == -2 || axisValue[i] == (shapeLens - 2)) {
                axisValue[i] = shapeLens - 1;
            }
        }

        ge::AttrUtils::SetListInt(newOpDescPtr, AXIS, axisValue);

        return SUCCESS;
    }

    Status softmaxTransFusionPass::INFuison(
        ge::ComputeGraph& graph,
        ge::NodePtr& inNodePtr, vector<ge::NodePtr>& newNodes)
    {
        ge::OpDescPtr inOpDescPtr = inNodePtr->GetOpDesc();
        FUSION_PASS_CHECK(inOpDescPtr == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                         inOpDescPtr->GetName().c_str()), return PARAM_INVALID);
        OP_LOGI(FUSED_OP_TYPE.c_str(), "NODE %s 1", inOpDescPtr->GetName().c_str());

        ge::GeTensorDesc xInputDesc = inOpDescPtr->GetInputDesc(0);
        vector<int64_t> inputShape = xInputDesc.GetShape().GetDims();
        FUSION_PASS_CHECK(inputShape.empty(),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's input shape is null, fusion failed.",
                         inOpDescPtr->GetName().c_str()), return PARAM_INVALID);

        uint32_t shapeLens = inputShape.size();

        vector<int32_t> axisValue;
        vector<int32_t>::iterator iterNegAxis, iterPosAxis;
        ge::AttrUtils::GetListInt(inOpDescPtr, AXIS, axisValue);
        iterNegAxis = find(axisValue.begin(), axisValue.end(), -1);
        iterPosAxis = find(axisValue.begin(), axisValue.end(), shapeLens - 1);

        if (shapeLens > 1 and (iterNegAxis != axisValue.end() or iterPosAxis != axisValue.end()) and
            inputShape.back() < 16 and inputShape[shapeLens - 1] * 10 < inputShape[shapeLens - 2] and
            inputShape[shapeLens - 2] < 600000) {

            // create softmax opdesc
            std::shared_ptr<ge::OpDesc> SoftmaxV2OpDescPtr = nullptr;
            SoftmaxV2OpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_new", "SoftmaxV2");

            FUSION_PASS_CHECK(SetAttrValueForNewNode(inOpDescPtr, SoftmaxV2OpDescPtr, shapeLens) != SUCCESS,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Update softmax attr failed."), return FAILED);

            // create transpose opdesc
            std::shared_ptr<ge::OpDesc> transposeOpDescPtr = nullptr;
            transposeOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_input", "TransposeD");

            std::shared_ptr<ge::OpDesc> transposeOutOpDescPtr = nullptr;
            transposeOutOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_out", "TransposeD");

            FUSION_PASS_CHECK(SetAttrValue(transposeOpDescPtr, shapeLens, 1) != SUCCESS,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "set transpose perm failed."), return FAILED);
            FUSION_PASS_CHECK(SetAttrValue(transposeOutOpDescPtr, shapeLens, 1) != SUCCESS,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "set transpose perm failed."), return FAILED);


            //get transpose input
            ge::GeTensorDesc tpsXInputTensorDesc = inOpDescPtr->GetInputDesc(0);

            // fill transpose output TensorDesc
            vector<int64_t> outShape;
            for (uint32_t i = 0; i < shapeLens; i++) {
                outShape.push_back(inputShape[i]);
            }

            int64_t tempSize = outShape.back();
            outShape[shapeLens - 1] = outShape[shapeLens - 2];
            outShape[shapeLens - 2] = tempSize;

            GeShape out_shape(outShape);

            ge::GeTensorDesc tpsyOutputTensorDesc;
            tpsyOutputTensorDesc = inOpDescPtr->GetOutputDesc(0);
            tpsyOutputTensorDesc.SetShape(out_shape);
            tpsyOutputTensorDesc.SetOriginShape(out_shape);
            tpsyOutputTensorDesc.SetFormat(tpsXInputTensorDesc.GetFormat());
            tpsyOutputTensorDesc.SetOriginFormat(FORMAT_ND);
            tpsyOutputTensorDesc.SetFormat(FORMAT_ND);

            OP_LOGI(FUSED_OP_TYPE.c_str(), "Set tpsyOutputTensorDesc shape Done");

            ge::GeTensorDesc tpsOutyOutputTensorDesc = inOpDescPtr->GetInputDesc(0);


            // get SoftmaxV2 output

            ge::GeTensorDesc yOutputTensorDesc;
            yOutputTensorDesc = inOpDescPtr->GetOutputDesc(0);
            yOutputTensorDesc.SetShape(tpsyOutputTensorDesc.GetShape());
            yOutputTensorDesc.SetOriginShape(tpsyOutputTensorDesc.GetShape());
            yOutputTensorDesc.SetFormat(tpsyOutputTensorDesc.GetFormat());
            yOutputTensorDesc.SetOriginFormat(FORMAT_ND);
            yOutputTensorDesc.SetFormat(FORMAT_ND);

            transposeOpDescPtr->AddInputDesc("x", tpsXInputTensorDesc);
            transposeOpDescPtr->AddOutputDesc("y", tpsyOutputTensorDesc);

            SoftmaxV2OpDescPtr->AddInputDesc("x", tpsyOutputTensorDesc);
            SoftmaxV2OpDescPtr->AddOutputDesc("y", yOutputTensorDesc);

            transposeOutOpDescPtr->AddInputDesc("x", yOutputTensorDesc);
            transposeOutOpDescPtr->AddOutputDesc("y", tpsOutyOutputTensorDesc);

            OP_LOGI(FUSED_OP_TYPE.c_str(), "Set SoftmaxV2OpDescPtr connect Done");

            // add tranposes and softmaxv2 node to graph
            ge::NodePtr transposeNodePtr = graph.AddNode(transposeOpDescPtr);
            ge::NodePtr SoftmaxV2NodePtr = graph.AddNode(SoftmaxV2OpDescPtr);
            ge::NodePtr transposeOutNodePtr = graph.AddNode(transposeOutOpDescPtr);
            newNodes.push_back(transposeNodePtr);
            newNodes.push_back(SoftmaxV2NodePtr);
            newNodes.push_back(transposeOutNodePtr);

            FUSION_PASS_CHECK(transposeNodePtr == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: transposeNodePtr is null, fusion failed."),
                     return FAILED);
            FUSION_PASS_CHECK(SoftmaxV2NodePtr == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: SoftmaxV2NodePtr is null, fusion failed."),
                     return FAILED);
            FUSION_PASS_CHECK(transposeOutNodePtr == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode: transposeOutNodePtr is null, fusion failed."),
                     return FAILED);

            // add the edge
            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                inNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
                transposeNodePtr->GetInDataAnchor(0)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from data node:%s to transpose node:%s failed.",
                             inNodePtr->GetInDataAnchor(0)
                                 ->GetPeerOutAnchor()
                                 ->GetOwnerNode()->GetName().c_str(),
                             transposeNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                transposeNodePtr->GetOutAnchor(0), SoftmaxV2NodePtr->GetInDataAnchor(0)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from outputedge node:%s to transpose node:%s failed.",
                             SoftmaxV2NodePtr->GetName().c_str(),
                             transposeNodePtr->GetName().c_str()),
                     return FAILED);

            FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(
                SoftmaxV2NodePtr->GetOutAnchor(0), transposeOutNodePtr->GetInDataAnchor(0)),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from outputedge node:%s to transpose node:%s failed.",
                             SoftmaxV2NodePtr->GetName().c_str(),
                             transposeOutNodePtr->GetName().c_str()),
                     return FAILED);

            // add the output of transpose edge
            size_t outanchorsize = inNodePtr->GetAllOutAnchors().size();
            for (size_t outindex = 0; outindex < outanchorsize; outindex++) {
                for (auto inDataAnchor : inNodePtr->GetOutDataAnchor(outindex)->GetPeerInDataAnchors()) {
                    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(inNodePtr->GetOutDataAnchor(outindex),
                                                        inDataAnchor) != SUCCESS,
                             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove SoftmaxV2 out data edge failed."), return FAILED);
                    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeOutNodePtr->GetOutDataAnchor(outindex),
                                                     inDataAnchor) != SUCCESS,
                             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add SoftmaxV2 out data edge failed."), return FAILED);
                }
            }

            // remove Normalize from graph
            FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(inNodePtr),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "remove inNodePtr node[%s] failed", inNodePtr->GetName().c_str()),
                     return FAILED);
            return SUCCESS;
        } else {
            return NOT_CHANGED;
        }
    }

    Status softmaxTransFusionPass::SetAttrValue(
        const ge::OpDescPtr& OpDescPtr, int64_t shapeLens, int32_t transfer)
    {

        // transfer the total image axis -2 and axis -1 for this case
        if (transfer == 1) {
            vector<int32_t> permValue;
            for (int32_t i = 0; i < shapeLens; i++) {
                permValue.push_back(i);
                OP_LOGI(FUSED_OP_TYPE.c_str(), "Set SetAttrValue %d", i);
            }

            //reversed aixs -2 and -1
            int temp = permValue.back();
            permValue[shapeLens - 1] = permValue[shapeLens - 2];
            permValue[shapeLens - 2] = temp;

            ge::AttrUtils::SetListInt(OpDescPtr, "perm", permValue);
            return SUCCESS;
        }

        return FAILED;
    }

    REGISTER_PASS("softmaxTransFusionPass", BUILT_IN_GRAPH_PASS, softmaxTransFusionPass);
}
