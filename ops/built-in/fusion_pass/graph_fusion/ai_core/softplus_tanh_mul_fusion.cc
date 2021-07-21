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
 * \file softplus_tanh_mul_fusion.cpp
 * \brief fuse softplus, tanh, mul to mish
 */
#include "softplus_tanh_mul_fusion.h"
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"


namespace fe {
    static const string PATTERN_SOFTPLUS = "softplus";
    static const string PATTERN_TANH = "tanh";
    static const string PATTERN_MUL = "mul";
    static const string PATTERN_MISH = "mish";
    static const string SOFTPLUS = "Softplus";
    static const string TANH = "Tanh";
    static const string MUL = "Mul";
    static const string MISH = "Mish";
    vector<FusionPattern*> SoftplusTanhMulPass::DefinePatterns() {
      vector<FusionPattern*> patterns;
      FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftplusTanhMulFusion");
      FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object not success."), return patterns);

    /*
           input0 ----
             |        |
          softplus    |               input0
             |        |      --->       |
            tanh      |                mish
             |        |                 |
            mul <-----                output0
             |
          output0
    */

      pattern->AddOpDesc(PATTERN_SOFTPLUS, {SOFTPLUS})
              .AddOpDesc(PATTERN_TANH, {TANH})
              .AddOpDesc(PATTERN_MUL, {MUL})
              .SetInputs(PATTERN_TANH, {PATTERN_SOFTPLUS})
              .SetInputs(PATTERN_MUL, {PATTERN_TANH})
              .SetOutput(PATTERN_MUL);
      patterns.push_back(pattern);
      return patterns;
    }

    Status SoftplusTanhMulPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
      ge::NodePtr softPlusNode = GetNodeFromMapping(PATTERN_SOFTPLUS, mapping);
      FUSION_PASS_CHECK(softPlusNode == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),"Node Softplus is null, fusion failed."),
                        return PARAM_INVALID);
      ge::NodePtr tanhNode = GetNodeFromMapping(PATTERN_TANH, mapping);
      FUSION_PASS_CHECK(tanhNode == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node Tanh is null, fusion failed."),
                        return PARAM_INVALID);
      ge::NodePtr mulNode = GetNodeFromMapping(PATTERN_MUL, mapping);
      FUSION_PASS_CHECK(mulNode == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node Mul is null, fusion failed."),
                        return PARAM_INVALID);

      // softplus and tanh must be single output
      if (softPlusNode->GetOutAllNodes().size() != 1 || tanhNode->GetOutAllNodes().size() != 1) {
        OP_LOGI("The softplus node or tanh node is not single output, will not fusion to mish.");
        return NOT_CHANGED;
      }

      // softplus and tanh input size must be 1, mul input size must be 2
      if (softPlusNode->GetInAllNodes().size() != 1 || tanhNode->GetInAllNodes().size() != 1 ||
          mulNode->GetInAllNodes().size() != 2) {
        OP_LOGI("The softplus or tanh node input size is not 1 or mul node input size is not 2");
        return NOT_CHANGED;
      }

      bool isMish = softPlusNode->GetInDataAnchor(0)->GetPeerOutAnchor() ==
                    mulNode->GetInDataAnchor(0)->GetPeerOutAnchor() ||
                    softPlusNode->GetInDataAnchor(0)->GetPeerOutAnchor() ==
                    mulNode->GetInDataAnchor(1)->GetPeerOutAnchor();
      if (!isMish) {
        OP_LOGI("The softplus node's input is equal to neither of the mul node's inputs, can not fusion to mish.");
        return NOT_CHANGED;
      }

      // create a new mish node
      OpDescPtr softPlusOpDescPtr = softPlusNode->GetOpDesc();
      std::shared_ptr<ge::OpDesc> mishOpDescPtr = nullptr;
      std::string newName = softPlusOpDescPtr->GetName() + tanhNode->GetName() + mulNode->GetName();
      FUSION_PASS_MAKE_SHARED(mishOpDescPtr = std::make_shared<ge::OpDesc>(newName, MISH), return NOT_CHANGED);

      // add input and output for mish node
      ge::GeTensorDesc inputTensorDesc = softPlusNode->GetOpDesc()->GetInputDesc(0);
      ge::GeTensorDesc outputTensorDesc = mulNode->GetOpDesc()->GetOutputDesc(0);
      mishOpDescPtr->AddInputDesc("x", inputTensorDesc);
      mishOpDescPtr->AddOutputDesc("y", outputTensorDesc);

      // add mish node to graph
      ge::NodePtr mishNode = graph.AddNode(mishOpDescPtr);
      fusionNodes.push_back(mishNode);
      FUSION_PASS_CHECK(mishNode == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "FusionNode: mishNode is null, fusion failed."),
                        return FAILED);

      // link mish node
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(softPlusNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                           mishNode->GetInDataAnchor(0)),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from node:%s to node:%s failed.",
                                softPlusNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                                mishNode->GetName().c_str()),
                        return FAILED);
      for (const auto& InDataAnchorPtr : mulNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(mulNode->GetOutDataAnchor(0), InDataAnchorPtr),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge from node:%s to node:%s failed.",
                                  mulNode->GetName().c_str(),
                                  InDataAnchorPtr->GetOwnerNode()->GetName().c_str()),
                          return FAILED);
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(mishNode->GetOutDataAnchor(0), InDataAnchorPtr),
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from node:%s to node:%s failed.",
                                  mishNode->GetName().c_str(),
                                  InDataAnchorPtr->GetOwnerNode()->GetName().c_str()),
                          return FAILED);
      }

      // remove softplus tanh mul
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(softPlusNode),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove softplus node[%s] failed", softPlusNode->GetName().c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(tanhNode),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove tanh node[%s] failed", tanhNode->GetName().c_str()),
                        return FAILED);
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(mulNode),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove mul node[%s] failed", mulNode->GetName().c_str()),
                        return FAILED);

      OP_LOGD(FUSED_OP_TYPE.c_str(), "SoftplusTanhMulPass successful.");
      return SUCCESS;
    }
    REGISTER_PASS("SoftplusTanhMulPass", BUILT_IN_GRAPH_PASS, SoftplusTanhMulPass);
}  // namespace fe
