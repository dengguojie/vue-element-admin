/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file extremum_grad_fusion_pass.h
 * \brief Fusion Pass for full structure of MaximumGrad/MinimumGrad(only Dx,
 *   only Dy, Dx & Dy) with/without sum
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EXTREMUM_GRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EXTREMUM_GRAD_FUSION_PASS_H_

#include <map>
#include <string>
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ExtremumGradFusionPass : public PatternFusionBasePass {
 public:
  Status Run(ge::ComputeGraph& graph) override;
  Status Run(ge::ComputeGraph& graph, OpsKernelInfoStorePtr opsKernelInfoStorePtr) override;

 protected:
  std::vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool CheckImplyType() const;
  bool MatchDx(ge::NodePtr nodeSelect, std::map<std::string, ge::NodePtr>& recordMap);

  bool MatchDy(ge::NodePtr nodeSelect, std::map<std::string, ge::NodePtr>& recordMap);

  Status RunOnePatternFusion(ge::ComputeGraph& graph, const ge::NodePtr& nodeEqual);

  Status DoFusion(ge::ComputeGraph& graph, const std::map<std::string, ge::NodePtr>& recordMap,
                  vector<ge::NodePtr>& fusionNodes);

  Status RemoveNode(ge::ComputeGraph& graph, const std::map<std::string, ge::NodePtr>& recordMap,
                    std::string patternName);

  ge::NodePtr CreateExtremumGradNode(ge::ComputeGraph& graph, ge::NodePtr nodeEqual, ge::NodePtr selectDxNode,
                                     ge::NodePtr selectDyNode, const std::map<std::string, ge::NodePtr>& recordMap);

  Status SetExtreMumGradOpDesc(ge::OpDescPtr equalOpDesc, ge::OpDescPtr selectOpDesc,
                               ge::OpDescPtr extreGradOpDesc) const;

  Status AdjustAnchor(ge::OutDataAnchorPtr dzInputAnchor, ge::NodePtr nodeEqual, ge::NodePtr extreGradNode,
                      ge::NodePtr outputDxNode, ge::NodePtr outputDyNode) const;

  Status ReplaceEdgeDst(ge::OutDataAnchorPtr src, ge::InDataAnchorPtr dst, ge::InDataAnchorPtr newDst) const;

  Status ReplaceEdgeSrc(ge::OutDataAnchorPtr src, ge::OutDataAnchorPtr newSrc, ge::InDataAnchorPtr dst) const;

  bool CheckAttrMatch(const std::map<string, ge::NodePtr>& recordMap);

  void SetExtemDataDumpAttr(const std::map<string, ge::NodePtr>& recordMap, vector<ge::NodePtr>& fusionNodes);

  bool CheckEqualOp(ge::NodePtr nodeEqual) const;

  bool CheckNameScope(const string& nameA, const string& nameB) const;

  bool CheckZeroConstantOp(ge::NodePtr nodeZeros) const;

  bool CheckSelectOp(const ge::NodePtr& nodeSelect, const ge::NodePtr& nodeEqual) const;

  bool CheckSameZeroNode(ge::NodePtr nodeZeros, const map<string, ge::NodePtr>& recordMap);

  bool CheckSumOp(ge::NodePtr nodeSum, ge::NodePtr nodeEqual) const;

  ge::NodePtr FindNodeInRecordMap(const map<string, ge::NodePtr>& recordMap, string key);
  Status RemoveInputEdges(ge::ComputeGraph& graph, const ge::NodePtr node) const;
  Status RemoveOutputEdges(ge::NodePtr node) const;
  const std::string CONSTANT = "Const";
  const string FUSED_OP_TYPE = "MaximumGrad/MinimumGrad";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_EXTREMUM_GRAD_FUSION_PASS_H_
