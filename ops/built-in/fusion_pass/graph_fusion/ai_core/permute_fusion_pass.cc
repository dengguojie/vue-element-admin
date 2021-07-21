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
 * \file permute_fusion_pass.cpp
 * \brief
 */
#include "permute_fusion_pass.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char FUSED_NODE[] = "Permute";
static const char TRANSPOSED_NODE[] = "TransposeD";
static const char PATTERN_FUSEDNODE[] = "Permute";
static const int NUM_0 = 0;
static const int NUM_1 = 1;
static const int NUM_2 = 2;
static const int NUM_3 = 3;
static const int NUM_4 = 4;

vector<FusionPattern*> PermuteFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PermuteFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);

  return patterns;
}

Status PermuteFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  ge::NodePtr permuteNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::NodePtr permuteNodeNew = nullptr;

  FUSION_PASS_CHECK(permuteNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "permute node is null"),
                    return PARAM_INVALID);
  ge::OpDescPtr permuteOpDesc = permuteNode->GetOpDesc();
  FUSION_PASS_CHECK(permuteOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "permuteOpDesc is null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc permuteInputOpDesc = permuteOpDesc->GetInputDesc(0);
  ge::GeTensorDesc permuteOutputOpDesc = permuteOpDesc->GetOutputDesc(0);
  ge::Format inputFormat = permuteInputOpDesc.GetFormat();
  ge::Format outputFormat = permuteOutputOpDesc.GetFormat();
  inputFormat = FORMAT_ND;
  outputFormat = FORMAT_ND;
  permuteOutputOpDesc.SetOriginFormat(outputFormat);
  permuteOutputOpDesc.SetFormat(outputFormat);
  permuteOpDesc->UpdateOutputDesc(0, permuteOutputOpDesc);
  permuteInputOpDesc.SetOriginFormat(inputFormat);
  permuteInputOpDesc.SetFormat(inputFormat);
  permuteOpDesc->UpdateInputDesc(0, permuteInputOpDesc);
  permuteOpDesc->SetType(TRANSPOSED_NODE);

  ge::GeShape inputShape = permuteInputOpDesc.GetShape();
  if (inputShape.GetDimNum() == NUM_4) {
    std::vector<int64_t> permList;
    ge::AttrUtils::GetListInt(permuteOpDesc, "perm", permList);
    bool checkPass = false;
    if (permList.size() == NUM_4 && permList[0] == NUM_0 && permList[1] == NUM_3 && permList[2] == NUM_2 &&
        permList[3] == NUM_1) {
      checkPass = true;
    }

    if (checkPass) {
      ge::OpDescPtr permuteOpDescNew = AttrUtils::CloneOpDesc(permuteOpDesc);
      permuteOpDescNew->SetName(permuteOpDesc->GetName() + "/New");

      // update the output shape and input shape of permuteOpDesc/permuteOpDescNew
      std::vector<int64_t> dimsIn = inputShape.GetDims();
      std::vector<int64_t> dimsInNew;
      dimsInNew.push_back(dimsIn[0]);
      dimsInNew.push_back(dimsIn[1]);
      dimsInNew.push_back(dimsIn[3]);
      dimsInNew.push_back(dimsIn[2]);

      ge::GeShape assistShape(dimsInNew);
      permuteOutputOpDesc.SetShape(assistShape);
      permuteOutputOpDesc.SetOriginShape(assistShape);
      ge::TensorUtils::SetRealDimCnt(permuteOutputOpDesc, NUM_4);
      permuteOpDesc->UpdateOutputDesc(0, permuteOutputOpDesc);
      permuteOpDescNew->UpdateInputDesc(0, permuteOutputOpDesc);

      // revise the attr of permuteOpDesc and permuteOpDescNew
      ge::AttrUtils::SetListInt(permuteOpDesc, "perm", {NUM_0, NUM_1, NUM_3, NUM_2});
      ge::AttrUtils::SetListInt(permuteOpDesc, "order", {NUM_0, NUM_1, NUM_3, NUM_2});
      ge::AttrUtils::SetListInt(permuteOpDescNew, "perm", {NUM_0, NUM_2, NUM_3, NUM_1});
      ge::AttrUtils::SetListInt(permuteOpDescNew, "order", {NUM_0, NUM_2, NUM_3, NUM_1});

      // add permuteOpDescNew to the graph
      permuteNodeNew = graph.AddNode(permuteOpDescNew);

      // connect the output 0 of permuteNodeNew to output 0 of permuteNode
      if (permuteNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
        for (InDataAnchorPtr inAnchorPtr : permuteNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
          inAnchorPtr->UnlinkAll();
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(permuteNodeNew->GetOutDataAnchor(0), inAnchorPtr),
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                    "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                                    permuteNodeNew->GetName().c_str(), permuteNode->GetName().c_str()),
                            return FAILED);
          OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
                  permuteNodeNew->GetName().c_str(), permuteNode->GetName().c_str());
        }
      }

      // connect the output 0 of permuteNode to input 0 of permuteNodeNew
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(permuteNode->GetOutDataAnchor(0), permuteNodeNew->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d] failed.",
                  permuteNode->GetName().c_str(), 0, permuteNodeNew->GetName().c_str(), 0),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d].",
              permuteNode->GetName().c_str(), 0, permuteNodeNew->GetName().c_str(), 0);

      // unlink all control output of permuteNode
      if (permuteNode->GetOutControlAnchor() != nullptr) {
        // connect the control output of permuteNodeNew to control output of permuteNode
        for (unsigned int i = 0; i < permuteNode->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(permuteNodeNew->GetOutControlAnchor(),
                                                 permuteNode->GetOutControlAnchor()->GetPeerInControlAnchors().at(i)),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge from fused node:%s's control index to fusion node:%s's control index[%d] failed.",
                      permuteNodeNew->GetName().c_str(), permuteNode->GetName().c_str(), i),
              return FAILED);
          OP_LOGD(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's control index to fusion node:%s's control index[%d].",
                  permuteNodeNew->GetName().c_str(), permuteNode->GetName().c_str(), i);
        }
        for (auto inControlAnchor : permuteNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(permuteNode->GetOutControlAnchor(), inControlAnchor),
                            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge from fused node:%s's output control failed.",
                                    permuteNode->GetName().c_str()),
                            return FAILED);
          OP_LOGD(FUSED_OP_TYPE.c_str(), "Remove edge from fused node:%s's output control index.",
                  permuteNode->GetName().c_str());
        }
      }

      newNodes.push_back(permuteNodeNew);
    }
  }

  return SUCCESS;
}

static const char PermutePassName[] = "PermuteFusionPass";
REGISTER_PASS(PermutePassName, BUILT_IN_GRAPH_PASS, PermuteFusionPass);
}  // namespace fe
