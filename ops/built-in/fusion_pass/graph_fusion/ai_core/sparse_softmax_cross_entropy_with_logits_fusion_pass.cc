/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file sparse_softmax_cross_entropy_with_logits_fusion_pass.cpp
 * \brief SparesSoftMax fusion pass(SparesSoftMax --> GatherV2 & SoftMax)
 */
#include "sparse_softmax_cross_entropy_with_logits_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
using namespace std;
using namespace ge;
namespace fe {
static const std::string CONSTANTOP = "Constant";

static const std::string PATTERN_SPARSE_SOFTMAX = "SparseSoftmaxCrossEntropyWithLogits";

static const char* SPARSE_SOFTMAX = "SparseSoftmaxCrossEntropyWithLogits";

vector<FusionPattern*> SparseSoftMaxFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SparseSoftMaxFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SPARSE_SOFTMAX, {SPARSE_SOFTMAX}).SetOutput(PATTERN_SPARSE_SOFTMAX);

  patterns.push_back(pattern);

  return patterns;
}
Status SparseSoftMaxFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_SPARSE_SOFTMAX, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  // Get the description (input, output, name, attribute) of the node in the original image
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc fusedDesc1 = fusedDesc->GetInputDesc(0);
  // Define auxiliary matrix shape
  ge::GeShape sparseSoftmaxInputShape = fusedDesc1.GetShape();
  // GESHAPE->vector
  vector<int64_t> dimInfo = sparseSoftmaxInputShape.GetDims();
  int32_t depth_size = dimInfo[1];
  int32_t labels_size = dimInfo[0];
  vector<int64_t> size_dim;
  size_dim.push_back(depth_size);

  // Add the description (input, output, name, attribute) of the Onehot node
  ge::OpDescPtr OneHot;
  std::string SparesSoftMaxName = fusedNode->GetOpDesc()->GetName();

  // Define node name----->SparseSoftmaxCrossEntropyWithLogits_OneHot
  OneHot = std::make_shared<ge::OpDesc>(SparesSoftMaxName + "_OneHot", "OneHotD");

  // Define out auxiliary matrix shape of OneHot node
  vector<int64_t> Onehot_out_shape;
  Onehot_out_shape.push_back(depth_size);
  Onehot_out_shape.push_back(depth_size);
  ge::GeShape assitShape0(Onehot_out_shape);

  // Set node properties
  ge::GeTensorDesc tensorDesc0(GeShape(), ge::FORMAT_ND, ge::DT_INT32);
  tensorDesc0.SetShape(assitShape0);
  int32_t realDimCnt0 = assitShape0.GetDimNum();
  ge::TensorUtils::SetRealDimCnt(tensorDesc0, realDimCnt0);
  OneHot->AddOutputDesc("y", tensorDesc0);
  ge::NodePtr OneHotNode = graph.AddNode(OneHot);
  fusionNodes.push_back(OneHotNode);
  FUSION_PASS_CHECK(
      OneHotNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "OneHotNode fusionNode:%s is null, fusion failed.", OneHotNode->GetName().c_str()),
      return PARAM_INVALID);

  // GeShape achieve shape size,size_dim is a vector
  ge::GeShape assitShape(size_dim);
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_INT32);
  tensorDesc.SetShape(assitShape);

  int32_t realDimCnt = assitShape.GetDimNum();
  ge::TensorUtils::SetRealDimCnt(tensorDesc, realDimCnt);

  ge::GeTensorPtr assitPtr = nullptr;
  // Construct pointer value of depth_size(input1 of onehot )
  unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[depth_size]);
  int32_t* ptr = inputAssit.get();
  for (int32_t i = 0; i < depth_size; i++) {
    *ptr = i;
    ptr++;
  }
  assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                            sizeof(int32_t) * depth_size);

  // Construct pointer value(input2 of onehot--->Default value 1)
  ge::GeTensorPtr assitPtr2 = nullptr;
  vector<int64_t> on_value;
  on_value.push_back(1);
  int32_t size_one = 1;
  ge::GeShape assitShape1(on_value);
  ge::GeTensorDesc tensorDesc1(GeShape(), ge::FORMAT_ND, ge::DT_INT32);
  tensorDesc1.SetShape(assitShape1);
  int32_t realDimCnt1 = assitShape1.GetDimNum();
  ge::TensorUtils::SetRealDimCnt(tensorDesc1, realDimCnt1);

  unique_ptr<int32_t[]> inputAssit1(new (std::nothrow) int32_t[size_one]);

  // Get original pointer
  int32_t* ptr1 = inputAssit1.get();
  *ptr1 = size_one;
  assitPtr2 = std::make_shared<ge::GeTensor>(tensorDesc1, reinterpret_cast<uint8_t*>(inputAssit1.get()),
                                             sizeof(int32_t) * size_one);

  // Construct pointer value(input3 of onehot--->Default value 0)
  ge::GeTensorPtr assitPtr3 = nullptr;
  vector<int64_t> off_value;
  off_value.push_back(1);
  int32_t size_zero = 0;
  ge::GeShape assitShape2(off_value);
  ge::GeTensorDesc tensorDesc2(GeShape(), ge::FORMAT_ND, ge::DT_INT32);
  tensorDesc2.SetShape(assitShape2);

  int32_t realDimCnt2 = assitShape2.GetDimNum();
  ge::TensorUtils::SetRealDimCnt(tensorDesc2, realDimCnt2);

  unique_ptr<int32_t[]> inputAssit2(new (std::nothrow) int32_t[size_zero]);

  // Get original pointer
  int32_t* ptr2 = inputAssit2.get();
  *ptr2 = size_zero;
  assitPtr3 = std::make_shared<ge::GeTensor>(tensorDesc2, reinterpret_cast<uint8_t*>(inputAssit2.get()),
                                             sizeof(int32_t) * size_zero);
  // set weight
  vector<ge::GeTensorPtr> weights = {assitPtr, assitPtr2, assitPtr3};
  ge::OpDescUtils::SetWeights(OneHotNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(OneHotNode);
  NodePtr constInput0 = constInputNodes[0];
  NodePtr constInput1 = constInputNodes[1];
  NodePtr constInput2 = constInputNodes[2];
  // SetType---->CONSTANTOP
  constInput0->GetOpDesc()->SetType(CONSTANTOP);
  constInput1->GetOpDesc()->SetType(CONSTANTOP);
  constInput2->GetOpDesc()->SetType(CONSTANTOP);
  ge::AttrUtils::SetInt(OneHot, "depth", depth_size);
  int32_t onehot_axis = -1;
  ge::AttrUtils::SetInt(OneHot, "axis", onehot_axis);
  ge::AttrUtils::SetStr(OneHot, "dtype", "int32");

  // Add the description (input, output, name, attribute) of the GatherV2 Node
  ge::OpDescPtr GatherV2;
  std::string SparesSoftMaxName1 = fusedNode->GetOpDesc()->GetName();
  GatherV2 = std::make_shared<ge::OpDesc>(SparesSoftMaxName1 + "_GatherV2", "GatherV2D");
  ge::GeTensorDesc GatherV2Desc1 = fusedDesc->GetInputDesc(1);
  vector<int64_t> gatherv2_out_shape;
  gatherv2_out_shape.push_back(labels_size);
  gatherv2_out_shape.push_back(depth_size);
  ge::GeShape assitShape3(gatherv2_out_shape);
  ge::GeTensorDesc tensorDesc3(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  tensorDesc3.SetShape(assitShape3);

  int32_t realDimCnt3 = assitShape3.GetDimNum();
  ge::TensorUtils::SetRealDimCnt(tensorDesc3, realDimCnt3);
  GatherV2->AddOutputDesc("output", tensorDesc3);

  ge::GeTensorDesc GatherV2Desc2 = OneHot->GetOutputDesc(0);
  GatherV2->AddInputDesc("x", GatherV2Desc2);
  GatherV2Desc1.SetDataType(ge::DT_INT32);
  GatherV2Desc1.SetFormat(ge::FORMAT_ND);
  GatherV2->AddInputDesc("indices", GatherV2Desc1);
  int32_t Gather_zero = 0;
  ge::AttrUtils::SetInt(GatherV2, "axis", Gather_zero);

  // CopyOpDesc(input, output, name, attribute) stay the same with fusedDesc
  ge::OpDescPtr SoftMax = AttrUtils::CopyOpDesc(fusedDesc);
  // GetInputDesc Modification not supported
  // MutableInputDesc Support modification
  SoftMax->MutableInputDesc(0)->SetFormat(ge::FORMAT_ND);
  SoftMax->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);
  SoftMax->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_ND);
  SoftMax->MutableInputDesc(1)->SetOriginShape(assitShape3);
  SoftMax->MutableInputDesc(1)->SetFormat(ge::FORMAT_ND);
  SoftMax->MutableInputDesc(1)->SetShape(assitShape3);
  SoftMax->MutableOutputDesc(1)->SetFormat(ge::FORMAT_ND);
  SoftMax->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT16);

  if (SoftMax == nullptr){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str());
    return PARAM_INVALID;
  }
  // OPTYPE
  SoftMax->SetType("SoftmaxCrossEntropyWithLogits");
  // ADD two node(GatherV2Node, SoftMaxNode)
  ge::NodePtr GatherV2Node = graph.AddNode(GatherV2);
  ge::NodePtr SoftMaxNode = graph.AddNode(SoftMax);
  fusionNodes.push_back(GatherV2Node);
  fusionNodes.push_back(SoftMaxNode);
  if (GatherV2Node == nullptr){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GatherV2Node fusionNode:%s is null, fusion failed.",
            GatherV2Node->GetName().c_str());
    return PARAM_INVALID;
  }
  if (SoftMaxNode == nullptr){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "SoftMaxNode fusionNode:%s is null, fusion failed.",
            SoftMaxNode->GetName().c_str());
    return PARAM_INVALID;
  }
  // Even side
  // Put OneHot Output0 side add to GatherV2 Input0 side
  if (SUCCESS != ge::GraphUtils::AddEdge(OneHotNode->GetOutDataAnchor(0), GatherV2Node->GetInDataAnchor(0))){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
            OneHotNode->GetName().c_str(), 0, GatherV2Node->GetName().c_str(), 0);
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
          OneHotNode->GetName().c_str(), 0, GatherV2Node->GetName().c_str(), 0);

  // Put origin Input1 side add to GatherV2 Input1 side
  if (SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(1)->GetPeerOutAnchor(), GatherV2Node->GetInDataAnchor(1))){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
            fusedNode->GetName().c_str(), 1, GatherV2Node->GetName().c_str(), 1);
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
          fusedNode->GetName().c_str(), 1, GatherV2Node->GetName().c_str(), 1);

  if (fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size() > 1) {
    int64_t Anchorssize = fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PeerOutControlAnchors Size:%d", Anchorssize);
    if (SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(1),
                                           GatherV2Node->GetInControlAnchor())){
      assitPtr = nullptr;
      assitPtr2 = nullptr;
      assitPtr3 = nullptr;
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
              fusedNode->GetName().c_str(), 1, GatherV2Node->GetName().c_str());
      return FAILED;
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), 1, GatherV2Node->GetName().c_str());
  }

  // Put origin Input0 side add to SoftMax Input0 side
  if (SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(1),
                                         GatherV2Node->GetInControlAnchor())){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
            fusedNode->GetName().c_str(), 0, SoftMaxNode->GetName().c_str(), 0);
    return FAILED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
          fusedNode->GetName().c_str(), 0, SoftMaxNode->GetName().c_str(), 0);

  if (fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size() > 0) {
    if (SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(0),
                                           SoftMaxNode->GetInControlAnchor())){
      assitPtr = nullptr;
      assitPtr2 = nullptr;
      assitPtr3 = nullptr;
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
              fusedNode->GetName().c_str(), 0, GatherV2Node->GetName().c_str());
      return FAILED;
    }
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), 0, GatherV2Node->GetName().c_str());
  }

  // Put GatherV2 Output0 side add to SoftMax Input1 side
  if (SUCCESS != ge::GraphUtils::AddEdge(GatherV2Node->GetOutDataAnchor(0), SoftMaxNode->GetInDataAnchor(1))){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
            GatherV2Node->GetName().c_str(), 0, SoftMaxNode->GetName().c_str(), 1);
    return FAILED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
          GatherV2Node->GetName().c_str(), 0, SoftMaxNode->GetName().c_str(), 1);

  // Get origin node Output0, put the node add to SoftMaxNode
  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of SPARSESOTMAX is [%d].",
            fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      if (SUCCESS != ge::GraphUtils::AddEdge(SoftMaxNode->GetOutDataAnchor(0), inAnchorPtr)){
        assitPtr = nullptr;
        assitPtr2 = nullptr;
        assitPtr3 = nullptr;
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st index failed.",
                fusedNode->GetName().c_str(), SoftMaxNode->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge1 from fused node:%s's 2nd index to fusion node:%s's 1st index.",
              fusedNode->GetName().c_str(), SoftMaxNode->GetName().c_str());
    }
  }

  if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of SPARSESOTMAX is [%d].",
            fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      if (SUCCESS != ge::GraphUtils::AddEdge(SoftMaxNode->GetOutDataAnchor(1), inAnchorPtr)){
        assitPtr = nullptr;
        assitPtr2 = nullptr;
        assitPtr3 = nullptr;
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st index failed.",
                fusedNode->GetName().c_str(), SoftMaxNode->GetName().c_str());
        return FAILED;
      }
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Add edge2 from fused node:%s's 2nd index to fusion node:%s's 1st index.",
              fusedNode->GetName().c_str(), SoftMaxNode->GetName().c_str());
    }
  }

  // To break off fusedNode all InControlAnchor
  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  // To break off fusedNode all InDataAnchors
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // delete  fusedNode
  if (ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode)){
    assitPtr = nullptr;
    assitPtr2 = nullptr;
    assitPtr3 = nullptr;
    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", fusedNode->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}
REGISTER_PASS("SparseSoftMaxFusionPass", BUILT_IN_GRAPH_PASS, SparseSoftMaxFusionPass);
}  // namespace fe
