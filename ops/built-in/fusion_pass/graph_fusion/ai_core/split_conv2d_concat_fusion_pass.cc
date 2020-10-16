/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief convert split+conv2d+concat to group conv2d
 *
 */
#include "split_conv2d_concat_fusion_pass.h"
#include <vector>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;

namespace fe {

static const string PATTERN_SPLIT = "split";
static const string PATTERN_CONV2D = "conv2d";
static const string PATTERN_CONCATV2 = "concat_v2";
static const string CONCATV2_TYPE = "ConcatV2";
static const string CONV2D_TYPE = "Conv2D";
static const string SPLIT_TYPE = "Split";
static const string CONST_TYPE = "Const";
static const string ATTR_GROUPS = "groups";
static const string ATTR_ORG_FMT = "origin_format";
static const string NAME_CCAT_DIM = "concat_dim";
static const string CCAT_HOST_OP = "Concatv2HostCpuOp";
static const string SPT_OUT_KEY = "y";
static const string CCAT_IN_KEY = "x";
static const std::set<string> NEW_CCAT_IN = {
    "Const", "QuantBiasOptimization", "QuantWeightRollBack",
    "QuantBiasRollBack"};
static const std::set<DataType> DATA_TYPE_IN = {
    DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32};

/* The graph struct need to adapt is shown as follows:
 *
 *               split
 *                 /\
 *         /    /       \    \
 *   conv2d conv2d ... conv2d conv2d
 *         \    \       /    /
 *                 \/
 *              concatv2
 *                 |
 *
 * Notice: the struct can be captured by
 *         split+conv2d+concat pattern
 */
vector<FusionPattern *> SplitConv2dConcatPass::DefinePatterns() {

    vector<FusionPattern *> patterns;
    FusionPattern *pattern =
        new (std::nothrow) FusionPattern("SplitConv2dConcatPass");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
             return patterns);
    pattern->AddOpDesc(PATTERN_SPLIT, {SPLIT_TYPE})
            .AddOpDesc(PATTERN_CONV2D, {CONV2D_TYPE})
            .AddOpDesc(PATTERN_CONCATV2, {CONCATV2_TYPE})
            .SetInputs(PATTERN_CONV2D, {PATTERN_SPLIT})
            .SetInputs(PATTERN_CONCATV2, {PATTERN_CONV2D})
            .SetOutput(PATTERN_CONCATV2);
    patterns.push_back(pattern);

    return patterns;
}

/* The processing flow is as follows:
 *
 *  1. check op type and inputs
 *
 * Notice: when clone a new OpDesc, it's essential to
 *         set right InputName and OuputName
 */
bool SplitConv2dConcatPass::AnalyzeMidLayer(ge::Node::Vistor<NodePtr> &sptOutput,
                     OpDescPtr &convGpDesc) {

    NodePtr aNode = sptOutput.at(0);
    auto aInput = aNode->GetInDataNodes();
    size_t aInCnt = aInput.size();
    FUSION_PASS_CHECK(aInCnt < 2, OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's inputs less than 2"),
             return false);
    OpDescPtr aDesc = aNode->GetOpDesc();
    GeTensorDesc aWTensor = aDesc->GetInputDesc(1);
    vector<int64_t> aWShape = aWTensor.GetOriginShape().GetDims();
    Format aWFormat = aWTensor.GetOriginFormat();
    FUSION_PASS_CHECK(aWFormat != FORMAT_HWCN && aWFormat != FORMAT_NCHW,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "weight format only support HWCN or NCHW"),
             return false);
    for (auto outNode : sptOutput) {
        std::string types = outNode->GetType();
        FUSION_PASS_CHECK(types != CONV2D_TYPE,
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's type should be %s, not %s",
                         CONV2D_TYPE.c_str(), types.c_str()),
                         return false);
        auto inputs = outNode->GetInDataNodes();
        size_t count = inputs.size();
        FUSION_PASS_CHECK(count != aInCnt,
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's inputs count is different"),
                 return false);
        auto outputs = outNode->GetOutDataNodes();
        FUSION_PASS_CHECK(outputs.at(0)->GetType() != CONCATV2_TYPE,
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "bottom layer is not concatv2"), return false);
        for (size_t j = 1; j < count; ++j) {
            FUSION_PASS_CHECK(NEW_CCAT_IN.find(inputs.at(j)->GetType()) ==
                        NEW_CCAT_IN.end(),
                     OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's other input is not const type"),
                     return false);
        }
        OpDescPtr desc = outNode->GetOpDesc();
        GeTensorDesc wTensor = desc->GetInputDesc(1);
        vector<int64_t> wShape = wTensor.GetOriginShape().GetDims();
        FUSION_PASS_CHECK(wShape != aWShape,
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's second input shape is different"),
                 return false);
        Format wFormat = wTensor.GetOriginFormat();
        FUSION_PASS_CHECK(wFormat != aWFormat,
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "middle layer's second input format is different"),
                 return false);
    }
    convGpDesc = AttrUtils::CloneOpDesc(aDesc);
    FUSION_PASS_CHECK(convGpDesc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "clone conv2d desc failed"),
             return false);
    convGpDesc->SetName(aDesc->GetName() + "/group_conv2d");
    auto inName = aDesc->GetAllInputName();
    auto outName = aDesc->GetAllOutputName();
    convGpDesc->UpdateInputName(inName);
    convGpDesc->UpdateOutputName(outName);

    return true;
}

/* The processing flow is as follows:
 *
 *  1. verify split and concat dim is input channel or not
 */
bool SplitConv2dConcatPass::VerifySptCcatAxis(OpDescPtr &convDesc, NodePtr &splitNode,
                       NodePtr &ccatNode) {

    auto sptInputs = splitNode->GetInDataNodes();
    NodePtr sptConst = sptInputs.at(0);
    FUSION_PASS_CHECK(sptConst->GetType() != CONST_TYPE,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "concat input 0 is not const node"),
             return false);
    OpDescPtr splitDesc = splitNode->GetOpDesc();
    size_t splitCnt = splitDesc->GetOutputsSize();
    size_t ccatCnt = 0;
    auto ccatInputs = ccatNode->GetInDataNodes();
    FUSION_PASS_CHECK(splitCnt != ccatInputs.size() - 1,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "concat input has no const node"),
             return false);
    for (auto inNode : ccatInputs) {
        if (inNode->GetType() == CONV2D_TYPE) {
            ccatCnt++;
        }
    }
    FUSION_PASS_CHECK(splitCnt != ccatCnt,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "split output count is not equal to concat input"),
             return false);

    vector<ge::GeTensorPtr> sptWeights =
        ge::OpDescUtils::MutableWeights(sptConst);
    FUSION_PASS_CHECK(sptWeights.size() < 1, OP_LOGW(FUSED_OP_TYPE.c_str(), "split weights get failed"),
             return false);
    ge::GeTensorPtr sptAxisPtr = sptWeights[0];
    FUSION_PASS_CHECK(sptAxisPtr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "split axis is nullptr"),
             return false);
    int32_t *sptAxis = (int32_t *) sptAxisPtr->GetData().data();
    FUSION_PASS_CHECK(sptAxis == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "sptAxis is nullptr"),
             return false);

    NodePtr ccatConst = ccatInputs.at(ccatInputs.size() - 1);
    vector<ge::GeTensorPtr> ccatWeights =
        ge::OpDescUtils::MutableWeights(ccatConst);
    FUSION_PASS_CHECK(ccatWeights.size() < 1, OP_LOGW(FUSED_OP_TYPE.c_str(), "concat weights get failed"),
             return false);
    ge::GeTensorPtr ccarAxisPtr = ccatWeights[0];
    FUSION_PASS_CHECK(ccarAxisPtr == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "concat axis is nullptr"),
             return false);
    int32_t *ccatAxis = (int32_t *) ccarAxisPtr->GetData().data();
    FUSION_PASS_CHECK(ccatAxis == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "ccatAxis is nullptr"),
             return false);
    FUSION_PASS_CHECK(ccatAxis[0] != sptAxis[0],
             OP_LOGW(FUSED_OP_TYPE.c_str(), "split axis is not equal to concat"),
             return false);

    GeTensorDesc xTensor = convDesc->GetInputDesc(0);
    std::string fmtStr = "";
    AttrUtils::GetStr(xTensor, ATTR_ORG_FMT, fmtStr);
    int32_t pos = fmtStr.find('C');
    FUSION_PASS_CHECK(ccatAxis[0] != pos,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "split axis is not on input channel"),
             return false);

    return true;
}

/* The processing flow is as follows:
 *
 *  1. get split input and concat output desc
 *  2. update all in/outputs desc of group conv2d
 */
bool SplitConv2dConcatPass::UpdateConv2dDesc(OpDescPtr &convDesc, NodePtr &splitNode,
                      NodePtr &ccatNode) {

    FUSION_PASS_CHECK(!VerifySptCcatAxis(convDesc, splitNode, ccatNode),
             OP_LOGW(FUSED_OP_TYPE.c_str(), "verify split and concat axis param failed"),
             return false);
    OpDescPtr splitDesc = splitNode->GetOpDesc();
    GeTensorDesc sInTensor = splitDesc->GetInputDesc(1);
    std::vector<int64_t> sInShape = sInTensor.GetOriginShape().GetDims();
    Format sFormat = sInTensor.GetOriginFormat();
    OpDescPtr ccatDesc = ccatNode->GetOpDesc();
    GeTensorDesc cOutTensor = ccatDesc->GetOutputDesc(0);
    std::vector<int64_t> cOutShape = cOutTensor.GetOriginShape().GetDims();
    Format cFormat = cOutTensor.GetOriginFormat();
    FUSION_PASS_CHECK(sFormat != cFormat,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "split format is not equal to concat"),
             return false);
    GeTensorDesc xTensor = convDesc->GetInputDesc(0);
    Format xFormat = xTensor.GetOriginFormat();
    FUSION_PASS_CHECK(sFormat != xFormat,
             OP_LOGW(FUSED_OP_TYPE.c_str(), "split format is not equal to conv2d"),
             return false);
    xTensor.SetShape(ge::GeShape(sInShape));
    xTensor.SetOriginShape(ge::GeShape(sInShape));
    convDesc->UpdateInputDesc(0, xTensor);
    GeTensorDesc yTensor = convDesc->GetOutputDesc(0);
    yTensor.SetShape(ge::GeShape(cOutShape));
    yTensor.SetOriginShape(ge::GeShape(cOutShape));
    convDesc->UpdateOutputDesc(0, yTensor);
    int64_t groups = splitDesc->GetOutputsSize();
    FUSION_PASS_CHECK(!AttrUtils::SetInt(convDesc, ATTR_GROUPS, groups),
             OP_LOGW(FUSED_OP_TYPE.c_str(), "set groups attr failed"),
             return false);
    size_t aInCnt = convDesc->GetInputsSize();
    for (size_t n = 1; n < aInCnt; ++n) {
        GeTensorDesc bTensor = convDesc->GetInputDesc(n);
        DataType bDtype = bTensor.GetDataType();
        FUSION_PASS_CHECK(DATA_TYPE_IN.find(bDtype) == DATA_TYPE_IN.end(),
                 OP_LOGW(FUSED_OP_TYPE.c_str(), "conv2d %d input data type only support float,"
                         " float16, int8 or int32",
                         int(n)),
                 return false);
        std::vector<int64_t> bInShape = bTensor.GetOriginShape().GetDims();
        size_t pos = 0;
        if (bInShape.size() == 4) {
            std::string fmtStr = "";
            AttrUtils::GetStr(bTensor, ATTR_ORG_FMT, fmtStr);
            size_t found = fmtStr.find('N');
            pos = found == std::string::npos ? 0 : found;
        }
        bInShape[pos] *= groups;
        bTensor.SetShape(ge::GeShape(bInShape));
        bTensor.SetOriginShape(ge::GeShape(bInShape));
        convDesc->UpdateInputDesc(n, bTensor);
    }

    return true;
}

/* The split and concatv2 in/outputs connection diagram:
 *
 *    split_dim    input
 *              \/
 *            split
 *              /\
 *      /  /  / ... \  \  \
 *     y0  y1 y2   y29 y30 y31
 *
 *     x0  x1 x2   x30 x31 concat_dim
 *      \  \  \ ... /  /  /
 *              \/
 *            concatv2
 *               |
 *             output
 *
 * Notice: all connected anchors may not be in order
 *
 * The processing flow is as follows:
 *
 *  1. create new concat nodes
 *  2. update it's in/output name and tensor desc
 */
bool SplitConv2dConcatPass::AddConcatDesc(NodePtr &splitNode, NodePtr &ccatNode,
                   std::vector<OpDescPtr> &constDesc) {

    OpDescPtr ccatDesc = ccatNode->GetOpDesc();
    auto outName = ccatDesc->GetAllOutputName();

    OpDescPtr sptDesc = splitNode->GetOpDesc();
    std::map<std::string, uint32_t> inName;
    auto sptOutName = sptDesc->GetAllOutputName();
    for (auto iter : sptOutName) {
        std::string key = iter.first;
        key.replace(key.find(SPT_OUT_KEY), CCAT_IN_KEY.length(), CCAT_IN_KEY);
        inName.insert(std::make_pair(key, iter.second));
    }
    uint32_t valueDim = sptOutName.size();
    inName.insert(std::make_pair(NAME_CCAT_DIM, valueDim));

    auto ccatInputs = ccatNode->GetInDataNodes();
    size_t count = ccatInputs.size() - 1;
    NodePtr aNode = ccatInputs.at(0);
    auto aInput = aNode->GetInDataNodes();
    size_t aInCnt = aInput.size();
    for (size_t coutIn = 1; coutIn < aInCnt; ++coutIn) {
        OpDescPtr nDesc = AttrUtils::CloneOpDesc(ccatDesc);
        FUSION_PASS_CHECK(nDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "clone concat desc failed"),
                 return false);
        nDesc->SetType(CCAT_HOST_OP);
        nDesc->UpdateInputName(inName);
        nDesc->UpdateOutputName(outName);
        nDesc->DelAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING);
        NodePtr aNNode = aInput.at(coutIn);
        OpDescPtr aNDesc = aNNode->GetOpDesc();
        GeTensorDesc aNTensor = aNDesc->GetOutputDesc(0);
        std::vector<int64_t> aNShape = aNTensor.GetOriginShape().GetDims();
        Format aFmt = aNTensor.GetOriginFormat();
        DataType aDtype = aNTensor.GetOriginDataType();
        size_t pos = 0;
        if (aNShape.size() == 4) {
            std::string fmtStr = "";
            AttrUtils::GetStr(aNTensor, ATTR_ORG_FMT, fmtStr);
            size_t found = fmtStr.find('N');
            pos = found == std::string::npos ? 0 : found;
        }
        for (size_t i = 0; i < count; ++i) {
            GeTensorDesc inDesc = nDesc->GetInputDesc(i);
            inDesc.SetShape(ge::GeShape(aNShape));
            inDesc.SetOriginShape(ge::GeShape(aNShape));
            inDesc.SetShape(ge::GeShape(aNShape));
            inDesc.SetFormat(aFmt);
            inDesc.SetOriginFormat(aFmt);
            inDesc.SetOriginDataType(aDtype);
            inDesc.SetDataType(aDtype);
            nDesc->UpdateInputDesc(i, inDesc);
        }
        aNShape[pos] *= int64_t(count);
        GeTensorDesc outDesc = nDesc->GetOutputDesc(0);
        outDesc.SetShape(ge::GeShape(aNShape));
        outDesc.SetOriginShape(ge::GeShape(aNShape));
        outDesc.SetFormat(aFmt);
        outDesc.SetOriginFormat(aFmt);
        outDesc.SetOriginDataType(aDtype);
        outDesc.SetDataType(aDtype);
        nDesc->UpdateOutputDesc(0, outDesc);
        std::string coutStr = std::to_string(coutIn);
        nDesc->SetName(aNDesc->GetName() + "/concat_" + coutStr);
        constDesc.push_back(nDesc);
    }

    return true;
}

/* The group conv2d fused with bn process of const concatv2:
 *
 *     x0  x1 x2   x30 x31 concat_dim
 *      \  \  \ ... /  /  /
 *              \/
 *        Concatv2HostCpuOp  <- [concatv2]
 *               |
 *         ConvBnFilterHost  <- [bn filter bias]
 *               |
 *          GroupPadding     <- [group pad]
 *               |
 *             output
 *
 * The processing flow is as follows:
 *
 *  1. unlink conv2d other inputs and link in
 *  2. create const_dim node and link in
 */
bool SplitConv2dConcatPass::LinkNewConcat(ge::ComputeGraph &graph, NodePtr &splitNode,
                   std::vector<NodePtr> &constCcat,
                   std::vector<NodePtr> &constDim) {

    auto sptOutAnchor = splitNode->GetAllOutDataAnchors();
    for(auto outAnchor : sptOutAnchor){
        int idx = outAnchor->GetIdx();
        InDataAnchorPtr aCAnchor = outAnchor->GetPeerInDataAnchors().at(0);
        NodePtr aCNode = aCAnchor->GetOwnerNode();
        size_t pos = 0;
        for (auto newCcat : constCcat) {
            InDataAnchorPtr newCcatIn = newCcat->GetInDataAnchor(idx);
            auto aCInAnchor = aCNode->GetInDataAnchor(++pos);
            auto aOutAnchor = aCInAnchor->GetPeerOutAnchor();
            aOutAnchor->Unlink(aCInAnchor);
            Status addRes = GraphUtils::AddEdge(
                                aOutAnchor,
                                newCcatIn);
            FUSION_PASS_CHECK(addRes != GRAPH_SUCCESS,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv2d other input failed"),
                     return false);
        }
    }
    NodePtr axisNode = splitNode->GetInDataNodes().at(0);
    OpDescPtr axisDesc = axisNode->GetOpDesc();
    size_t count = 0;
    for (auto newCcat : constCcat) {
        OpDescPtr ccatDesc = newCcat->GetOpDesc();
        int idx = ccatDesc->GetInputIndexByName(NAME_CCAT_DIM);
        InDataAnchorPtr lastInAnchor = newCcat->GetInDataAnchor(idx);
        OpDescPtr lDesc = AttrUtils::CloneOpDesc(axisDesc);
        FUSION_PASS_CHECK(lDesc == nullptr,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "clone concat last input desc failed"),
                 return false);
        std::string coutStr = std::to_string(count++);
        lDesc->SetName(axisDesc->GetName() + "/last_" + coutStr);
        NodePtr lNode = graph.AddNode(lDesc);

        GeTensorDesc ccatTensor = ccatDesc->GetInputDesc(0);
        std::vector<int64_t> cInShape = ccatTensor.GetOriginShape().GetDims();
        int32_t pos = 0;
        if (cInShape.size() == 4) {
            std::string fmtStr = "";
            AttrUtils::GetStr(ccatTensor, ATTR_ORG_FMT, fmtStr);
            size_t found = fmtStr.find('N');
            pos = found == std::string::npos ? 0 : found;
        }
        std::vector<int32_t> axis;
        axis.push_back(pos);
        vector<ge::GeTensorPtr> axisWeights =
        ge::OpDescUtils::MutableWeights(lNode);
        FUSION_PASS_CHECK(axisWeights.size() < 1, OP_LOGE(FUSED_OP_TYPE.c_str(), "axis weights get failed"),
                 return false);
        ge::GeTensorPtr axisPtr = axisWeights[0];
        FUSION_PASS_CHECK(axisPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "axis ptr is nullptr"),
                 return false);
        axisPtr->SetData(reinterpret_cast<uint8_t *>(axis.data()),
                         axis.size() * sizeof(int32_t));
        Status addRes = GraphUtils::AddEdge(
                            lNode->GetOutDataAnchor(0),
                            lastInAnchor);
        FUSION_PASS_CHECK(addRes != GRAPH_SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from const dim node to new concat failed"),
                 return false);
        constDim.push_back(lNode);
    }

    return true;

}

/* The processing flow is as follows:
 *
 *  1. unlink split and concat node
 *  2. link group conv2d with split previous and concat next node
 */
bool SplitConv2dConcatPass::LinkGroupConv2d(NodePtr &groupConv, NodePtr &splitNode, NodePtr &ccatNode,
                     std::vector<NodePtr> &constCcat) {

    OutDataAnchorPtr preAnchor =
        splitNode->GetInDataAnchor(1)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preAnchor == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "split input anchor is null"),
             return false);
    preAnchor->UnlinkAll();
    InDataAnchorPtr xAnchor = groupConv->GetInDataAnchor(0);
    FUSION_PASS_CHECK(xAnchor == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "group conv2d input anchor is null"),
             return false);
    FUSION_PASS_CHECK(GraphUtils::AddEdge(preAnchor, xAnchor) != GRAPH_SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from split input to conv2d failed"),
             return false);
    OutDataAnchorPtr ccatOut = ccatNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(ccatOut == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "concat output anchor is null"),
             return false);
    OutDataAnchorPtr yAnchor = groupConv->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(yAnchor == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "group conv2d output anchor is null"),
             return false);
    auto peerInAnchor =  ccatOut->GetPeerInDataAnchors();
    ccatOut->UnlinkAll();
    for (auto nextAnchor : peerInAnchor) {
        nextAnchor->UnlinkAll();
        FUSION_PASS_CHECK(GraphUtils::AddEdge(yAnchor, nextAnchor) != GRAPH_SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv2d output to concat next failed"),
                 return false);
    }
    size_t pos = 0;
    for (auto newCcat : constCcat) {
        Status addRes = GraphUtils::AddEdge(
                            newCcat->GetOutDataAnchor(0),
                            groupConv->GetInDataAnchor(++pos));
        FUSION_PASS_CHECK(addRes != GRAPH_SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from conv2d to new concat failed"),
                 return false);
    }

    return true;
}

/* The graph struct after adapt is shown as follows:
 *
 *          weight...weight    bias(if exist)
 *              \  |  /        /
 *    inputs   concatv2    concatv2
 *       \        |        /
 *        conv2d(with group)
 *                |
 *
 * Notice: weight or bias input should be concat by new concat node, while
 *         input image data leave as is, a conv2d node should be created
 *         with new shape and groups attr, after new conv2d node linked in
 *         the graph, other nodes should be deleted.
*/
Status SplitConv2dConcatPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                     std::vector<NodePtr> &newNodes) {

    OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter SplitConv2dConcatPass.");
    NodePtr splitNode = GetNodeFromMapping(PATTERN_SPLIT, mapping);
    NodePtr ccatNode = GetNodeFromMapping(PATTERN_CONCATV2, mapping);
    auto sptOutput = splitNode->GetOutDataNodes();

    OpDescPtr convGpDesc;
    FUSION_PASS_CHECK(!AnalyzeMidLayer(sptOutput, convGpDesc),
             OP_LOGW(FUSED_OP_TYPE.c_str(), "nothing changed on the graph"),
             return NOT_CHANGED);

    FUSION_PASS_CHECK(!UpdateConv2dDesc(convGpDesc, splitNode, ccatNode),
             OP_LOGW(FUSED_OP_TYPE.c_str(), "update group conv2d desc failed"),
             return NOT_CHANGED);
    NodePtr groupConv = graph.AddNode(convGpDesc);
    newNodes.push_back(groupConv);

    std::vector<OpDescPtr> constDesc;
    FUSION_PASS_CHECK(!AddConcatDesc(splitNode, ccatNode, constDesc),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "create conv2d other input concat desc failed"),
             return FAILED);
    std::vector<NodePtr> constCcat;
    for (auto newDesc : constDesc) {
        NodePtr newCcat = graph.AddNode(newDesc);
        constCcat.push_back(newCcat);
        newNodes.push_back(newCcat);
    }

    std::vector<NodePtr> constDim;
    FUSION_PASS_CHECK(!LinkNewConcat(graph, splitNode, constCcat, constDim),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "create concat last const node failed"),
             return FAILED);
    for (auto newLast : constDim) {
        newNodes.push_back(newLast);
    }

    FUSION_PASS_CHECK(!LinkGroupConv2d(groupConv, splitNode, ccatNode, constCcat),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "link group conv2d node and new nodes failed"),
             return FAILED);

    auto sptInputs = splitNode->GetInDataNodes();
    FUSION_PASS_CHECK(graph.RemoveNode(sptInputs.at(0)) != GRAPH_SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "remove split const input node failed"),
             return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(splitNode) != GRAPH_SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "remove split node failed"),
             return FAILED);
    for (auto ccatInNode : ccatNode->GetInDataNodes()) {
        FUSION_PASS_CHECK(graph.RemoveNode(ccatInNode) != GRAPH_SUCCESS,
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "remove unused concat input node failed"),
                 return FAILED);
    }
    FUSION_PASS_CHECK(graph.RemoveNode(ccatNode) != GRAPH_SUCCESS,
         OP_LOGE(FUSED_OP_TYPE.c_str(), "remove concat node failed"),
         return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave SplitConv2dConcatPass.");

    return SUCCESS;
}

REGISTER_PASS("ASplitConv2dConcatPass", BUILT_IN_GRAPH_PASS,
              SplitConv2dConcatPass);
}
