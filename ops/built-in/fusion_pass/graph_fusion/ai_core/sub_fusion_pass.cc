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
 * \file sub_fusion_pass.cc
 * \brief
 */
#include "sub_fusion_pass.h"

#include <vector>
#include <memory>

#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator_factory.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_SUB = "Sub";
vector<FusionPattern*> SubFusionPass::DefinePatterns() {
  OP_LOGI(PATTERN_SUB.c_str(), "Define SubFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("SubFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(PATTERN_SUB.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_SUB, {"Sub"}).SetOutput(PATTERN_SUB);
  patterns.push_back(pattern);
  OP_LOGI(PATTERN_SUB.c_str(), "Define SubFusionPass pattern end");
  return patterns;
}

Status SubFusionPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& newNodes) {
  OP_LOGI(PATTERN_SUB.c_str(), "Define SubFusionPass fusion begin.");
  NodePtr subNode = GetNodeFromMapping(PATTERN_SUB, mapping);

  FUSION_PASS_CHECK(subNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(PATTERN_SUB.c_str(), "Sub is null, fusion failed."),
                    return PARAM_INVALID);
  // two situations
  //                    op node                                op_node1            op_node2
  //  -------              |                     ---------        |                   |
  // | fused |             |                    | unfused |       |                   |
  //  -------       out data anchor              ---------  out data anchor      out data anchor
  //                /            \                                |                   |
  //               /              \                ======>        |                   |
  //      input data anchor   input data anchor            input data anchor    input data anchor
  //               \              /                                \                  /
  //                \            /                                  \                /
  //                    sub node                                         sub node
  //                       |                                                 |
  //                       |                                                 |
  //                  output anchor                                     output anchor
  // get input anchor of subnode
  auto inFirstAnchor = subNode->GetInDataAnchor(0);
  auto inSecondAnchor = subNode->GetInDataAnchor(1);

  // get related output anchor of output
  auto outFirstAnchor = inFirstAnchor->GetPeerOutAnchor();
  auto outSecondAnchor = inSecondAnchor->GetPeerOutAnchor();

  // get output anchor id
  int outFirstIdx = outFirstAnchor->GetIdx();
  int outSecondIdx = outSecondAnchor->GetIdx();
  OP_LOGI(PATTERN_SUB.c_str(), "outFirstIdx num is: %d", outFirstIdx);
  OP_LOGI(PATTERN_SUB.c_str(), "outSecondIdx num is: %d", outSecondIdx);

  // get related node of output anchor
  NodePtr getFirstNode = outFirstAnchor->GetOwnerNode();
  NodePtr getSecondNode = outSecondAnchor->GetOwnerNode();

  // sub's 2 inputs from diff nodes
  if (getFirstNode->GetName() != getSecondNode->GetName()) {
    // transfer to ub fusion
    OP_LOGI(PATTERN_SUB.c_str(), "SubFusionPass not changed");
    return NOT_CHANGED;
  }

  // outanchor from same node but ids is diff
  if (outFirstIdx != outSecondIdx) {
    // transfer to ub fusion
    OP_LOGI(PATTERN_SUB.c_str(), "SubFusionPass not changed");
    return NOT_CHANGED;
  }

  // replace sub -> const
  auto subOpDesc = subNode->GetOpDesc();
  GeTensorDesc subInputShapeTensor = subOpDesc->GetInputDesc(0);
  GeShape subShape = subInputShapeTensor.GetShape();
  DataType subType = subInputShapeTensor.GetDataType();

  vector<int64_t> subDimInfo = subShape.GetDims();
  int32_t shapeSize = 1;
  for (auto it : subDimInfo) {
    if(it < 0){
      OP_LOGI(PATTERN_SUB.c_str(), "SubFusionPass do not support dynamic shape, not changed");
      return NOT_CHANGED;
    }
    shapeSize *= static_cast<int32_t>(it);
  }
  vector<int32_t> zeroNum(shapeSize, 0);

  // set const node, format, dtype
  GeShape constShape = subShape;
  auto constDesc = GeTensorDesc(constShape, FORMAT_ND, subType);

  GeTensorPtr constDescTensor = nullptr;
  constDescTensor = make_shared<GeTensor>(constDesc);
  if (subType == DT_INT32) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(int32_t));
  } else if (subType == DT_FLOAT16) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(uint16_t));
  } else if (subType == DT_FLOAT) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(float));
  } else if (subType == DT_UINT8) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(uint8_t));
  } else if (subType == DT_INT8) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(int8_t));
  } else if (subType == DT_UINT16) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(uint16_t));
  } else if (subType == DT_INT16) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(int16_t));
  } else if (subType == DT_INT64) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(int64_t));
  } else if (subType == DT_DOUBLE) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(double));
  } else if (subType == DT_COMPLEX64) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()), zeroNum.size() * sizeof(int64_t));
  } else if (subType == DT_COMPLEX128) {
    constDescTensor->SetData(reinterpret_cast<uint8_t*>(zeroNum.data()),
                             zeroNum.size() * (sizeof(int64_t) + sizeof(int64_t)));
  } else {
    OP_LOGI(PATTERN_SUB.c_str(),
            "SubFusionPass type only support DT_INT32 DT_FLOAT16 DT_FLOAT DT_UINT8 DT_INT8 DT_UINT16 DT_INT64 "
            "DT_DOUBLE DT_COMPLEX64 DT_COMPLEX128");
    OP_LOGI(PATTERN_SUB.c_str(), "SubFusionPass not changed");
    return NOT_CHANGED;
  }

  OpDescPtr constOpDesc = OpDescUtils::CreateConstOp(constDescTensor);
  // produce const node
  NodePtr constNode = graph.AddNode(constOpDesc);

  // break input edge
  //
  //                 op node
  //                   |
  //                   |                                     op node           constant
  //            out data anchor                                 |                  |
  //            /             \                                 |                  |
  //           /               \                ========>       |                  |
  //   input data anchor   input data anchor                    |            out data anchor
  //           \               /                          out data anchor          |
  //            \             /                                                    |
  //               sub node
  auto constNodeInAnchor = constNode->GetInDataAnchor(0);
  GraphUtils::RemoveEdge(outFirstAnchor, inFirstAnchor);
  GraphUtils::RemoveEdge(outSecondAnchor, inSecondAnchor);

  // rebuild output edge
  auto outAnchor = subNode->GetOutDataAnchor(0);
  auto constOutAnchor = constNode->GetOutDataAnchor(0);
  for (auto outAnchorPeerIn : outAnchor->GetPeerInDataAnchors()) {
    GraphUtils::RemoveEdge(outAnchor, outAnchorPeerIn);
    GraphUtils::AddEdge(constOutAnchor, outAnchorPeerIn);
  }

  // remove aborted anchor
  inFirstAnchor->UnlinkAll();
  inSecondAnchor->UnlinkAll();
  outAnchor->UnlinkAll();

  // remove sub node
  if (graph.RemoveNode(subNode) != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(PATTERN_SUB.c_str(), "Failed to remove sub node.");
    return FAILED;
  }
  OP_LOGI(PATTERN_SUB.c_str(), "Define SubFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("SubFusionPass", BUILT_IN_GRAPH_PASS, SubFusionPass);
}  // namespace fe