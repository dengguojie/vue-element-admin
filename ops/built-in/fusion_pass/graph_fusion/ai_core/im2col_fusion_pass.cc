/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file im2col_fusion_pass.cpp
 * \brief
 */
#include "im2col_fusion_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "Im2col";
static const std::string PATTERN_FUSEDNODE = "Im2col";


vector<FusionPattern*> Im2colFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Im2colFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

static void AssistInit(const vector<int64_t>& const_vec, int32_t* output) {
  for (int32_t i = 0; i < const_vec.size(); ++i) {
    output[i] = const_vec[i];
  }
}

Status Im2colFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Im2colFusionPass.");
  ge::NodePtr im2colNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(im2colNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "im2col node is null."),
                    return PARAM_INVALID);              
  ge::OpDescPtr im2colOpDesc = im2colNode->GetOpDesc();
  FUSION_PASS_CHECK(im2colOpDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "im2col is null."), return PARAM_INVALID);

  ge::GeTensorDesc im2colInputOpDesc = im2colOpDesc->GetInputDesc(0);
  ge::GeTensorDesc im2colOutputOpDesc = im2colOpDesc->GetOutputDesc(0);
  ge::Format inputOriginFormat = im2colInputOpDesc.GetOriginFormat();
  ge::Format outputOriginFormat = im2colOutputOpDesc.GetOriginFormat();

  ge::GeShape im2colOutputShape = im2colOutputOpDesc.GetOriginShape();
  vector<int64_t> outDimInfo = im2colOutputShape.GetDims();
  int64_t outputN = 0;
  int64_t outputC = 0;
  int64_t outputH = 0;
  int64_t outputW = 0;

  if (outDimInfo.size() == 4 && outputOriginFormat == ge::FORMAT_NCHW) {
    outputN = outDimInfo[0];
    outputC = outDimInfo[1];
    outputH = outDimInfo[2];
    outputW = outDimInfo[3];
  } else {
    return NOT_CHANGED;
  }

  if (outputOriginFormat == ge::FORMAT_NCHW) {
    vector<int64_t> ksizes;
    ge::AttrUtils::GetListInt(im2colOpDesc, "ksizes", ksizes);
    FUSION_PASS_CHECK(ksizes.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "ksizes is null, please check!"), return FAILED);
    int64_t k = ksizes[0] * ksizes[1];
    vector<int64_t> reshape1DimInfo = {outputN, k, outputC / k, outputH, outputW};
    vector<int64_t> outputDimInfo = {outputN, outputC / k, k, outputH, outputW};
    vector<int64_t> reshape2DimInfo = {outputN, outputC, outputH, outputW};

    // creat node
    ge::OutDataAnchorPtr im2colAnchorPtr1 = im2colNode->GetOutDataAnchor(0);
    ge::NodePtr postNode = nullptr;
    auto reshape2DpPtr = im2colAnchorPtr1->GetPeerInDataAnchors().at(0);

    std::shared_ptr<ge::OpDesc> reshapeDesc1 = nullptr;
    reshapeDesc1 = std::make_shared<ge::OpDesc>(im2colNode->GetName() + "_reshape1_layer", "Reshape");
    FUSION_PASS_CHECK(reshapeDesc1 == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "reshape1 is null, Reshape failed."), return PARAM_INVALID);
    std::shared_ptr<ge::OpDesc> transposeDDesc1 = nullptr;
    transposeDDesc1 = std::make_shared<ge::OpDesc>(im2colNode->GetName() + "_transposeD_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc1 == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "transposeD is null, TransposeD failed."), return PARAM_INVALID);
    std::shared_ptr<ge::OpDesc> reshapeDesc2 = nullptr;
    reshapeDesc2 = std::make_shared<ge::OpDesc>(im2colNode->GetName() + "_reshape2_layer", "Reshape");
    FUSION_PASS_CHECK(reshapeDesc2 == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "reshape1 is null, Reshape failed."), return PARAM_INVALID);

    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[5]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[4]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    AssistInit(reshape1DimInfo, input_assist_1.get());
    AssistInit(reshape2DimInfo, input_assist_2.get());
    
    // geneate is_input_const
    vector<bool> is_input_const;
    is_input_const.push_back(false);
    is_input_const.push_back(true);

    // add input, output
    ge::GeTensorDesc input_desc1 = im2colNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(reshapeDesc1->AddInputDesc("x", input_desc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add reshapeDesc1 input failed."), return FAILED);
    ge::GeShape assitShape1(reshape1DimInfo);
    ge::GeShape assitShapeOrigin1(reshape1DimInfo);
    input_desc1.SetShape(assitShape1);
    input_desc1.SetOriginShape(assitShapeOrigin1);
    FUSION_PASS_CHECK(reshapeDesc1->AddOutputDesc("y", input_desc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add reshapeDesc1 output failed."), return FAILED);
    // add node
    ge::NodePtr reshapeNode1 = graph.AddNode(reshapeDesc1);

    // generate assist
    GeTensorDesc assist_desc;
    assist_desc.SetDataType(DT_INT32);
    assist_desc.SetFormat(FORMAT_ND);
    GeTensorPtr assist_ptr_1 = nullptr;
    GeTensorPtr assist_ptr_2 = nullptr;

    // add const
    assist_desc.SetShape(GeShape({5}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                            assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 5 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshapeNode1, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshapeNode1);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshapeDesc1->SetIsInputConst(is_input_const);

    // add input, output
    FUSION_PASS_CHECK(transposeDDesc1->AddInputDesc(input_desc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 input failed."), return FAILED);
    ge::GeShape assitShape(outputDimInfo);
    ge::GeShape assitShapeOrigin(outputDimInfo);
    input_desc1.SetShape(assitShape);
    input_desc1.SetOriginShape(assitShapeOrigin);
    FUSION_PASS_CHECK(transposeDDesc1->AddOutputDesc(input_desc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 output failed."), return FAILED);

    // add node
    ge::NodePtr transposeDNode1 = graph.AddNode(transposeDDesc1);
    ge::AttrUtils::SetListInt(transposeDDesc1, "perm", {0, 2, 1, 3, 4});

    // reshape2 add input and output
    ge::GeTensorDesc transOutputOpDesc = transposeDDesc1->GetOutputDesc(0);
    ge::GeTensorDesc output_desc1 = reshape2DpPtr->GetOwnerNode()->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(reshapeDesc2->AddInputDesc("x", transOutputOpDesc) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add reshapeDesc2 input failed."), return FAILED);
    FUSION_PASS_CHECK(reshapeDesc2->AddOutputDesc("y", output_desc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add reshapeDesc2 output failed."), return FAILED);

    ge::GeTensorDesc reshape2OutOpDesc = reshapeDesc2->GetOutputDesc(0);
    ge::GeShape assitShape2(reshape2DimInfo);
    ge::GeShape assitShapeOrigin2(reshape2DimInfo);
    reshape2OutOpDesc.SetShape(assitShape2);
    reshape2OutOpDesc.SetOriginShape(assitShapeOrigin2);
    reshape2OutOpDesc.SetFormat(ge::FORMAT_NCHW);
    reshape2OutOpDesc.SetOriginFormat(ge::FORMAT_NCHW);
    Status ret = reshapeDesc2->UpdateOutputDesc(0, reshape2OutOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "reshape2 UpdateOutputDesc failed."), return FAILED);

    // add node
    ge::NodePtr reshapeNode2 = graph.AddNode(reshapeDesc2);

    // add const
    assist_desc.SetShape(GeShape({4}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                            assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 4 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshapeNode2, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshapeNode2);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshapeDesc2->SetIsInputConst(is_input_const);


    // add edge between im2col and reshape1, reshape1 and transpose, transpose and reshape2 
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(im2colAnchorPtr1, reshapeNode1->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              im2colNode->GetName().c_str(), reshapeNode1->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshapeNode1->GetOutDataAnchor(0), transposeDNode1->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              reshapeNode1->GetName().c_str(), transposeDNode1->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode1->GetOutDataAnchor(0), reshapeNode2->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              transposeDNode1->GetName().c_str(), reshapeNode2->GetName().c_str()),
                      return FAILED);

    for (auto postAnchorPtr0 : im2colAnchorPtr1->GetPeerInDataAnchors()) {
      if (postAnchorPtr0->GetOwnerNode()->GetName() != im2colNode->GetName() + "_reshape1_layer") {
        postNode = postAnchorPtr0->GetOwnerNode();
        // remove edge between im2col and next node
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, im2colAnchorPtr1) != SUCCESS,
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between im2col and next node failed!"),
                          return FAILED);

        // add edge between reshape2 and post
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(reshapeNode2->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                  reshapeNode2->GetName().c_str(), im2colNode->GetName().c_str()),
                          return FAILED);
      }
    }
    return SUCCESS;
  } else {
    return NOT_CHANGED;
  }
}

REGISTER_PASS("Im2colFusionPass", BUILT_IN_GRAPH_PASS, Im2colFusionPass);
}  // namespace fe
