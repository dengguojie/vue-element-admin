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
 * \file deep_md_fusion_pass_util.h
 * \brief Deep MD fusion pass util
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DEEP_MD_FUSION_PASS_UTIL_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DEEP_MD_FUSION_PASS_UTIL_H_

#include <string>
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DeepMdFusionPassUtil {
 public:
  /**
   * Check value of split.
   * Expect value is split_count := 1, split_index := 0.
   * @param fusedOpType
   * @param node: Graph node which contains attribute: split_count, split_index.
   * @return status
   */
  static Status CheckSplitInitInfo(const std::string& fusedOpType, const ge::NodePtr& node);

  /**
   * Split node to AI Core and Vector Core.
   * @param fusedOpType
   * @param graph
   * @param fusedNode
   * @param nodeAic
   * @param nodeVec
   * @return status
   */
  static Status SplitNodeToAICoreAndVectorCore(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                               const ge::NodePtr& fusedNode, ge::NodePtr& nodeAic,
                                               ge::NodePtr& nodeVec);

  /**
   * Create add node.
   * @param fusedOpType
   * @param graph
   * @param addNode
   * @param addNodeName
   * @param preNodes
   * @param preNodeOutputIdx
   * @return status
   */
  static Status CreateAddNodeAfterSplitNode(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                            ge::NodePtr& addNode, const std::string& addNodeName,
                                            vector<ge::NodePtr>& preNodes, const uint32_t& preNodeOutputIdx);

  /**
   * Create ConcatD node.
   * @param fusedOpType
   * @param graph
   * @param concatNode
   * @param concatNodeName
   * @param preNodes
   * @param preNodeOutputIdx
   * @return status
   */
  static Status CreateConcatNodeAfterSplitNode(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                               ge::NodePtr& concatNode, const std::string& concatNodeName,
                                               const vector<ge::NodePtr>& preNodes, const uint32_t& preNodeOutputIdx,
                                               const vector<int32_t>& concatAttrs);

  /**
   * Clear fused node.
   * @param fusedOpType
   * @param graph
   * @param node
   * @return status
   */
  static Status ClearFusedNode(const std::string& fusedOpType, ge::ComputeGraph& graph, ge::NodePtr& node);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DEEP_MD_FUSION_PASS_UTIL_H_
