/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Fusion Pass for full structure of MaximumGrad/MinimumGrad(only Dx,
 * only Dy, Dx & Dy) with/without sum
 */

#ifndef FE_OPTIMIZER_FUSION_FULL_EXTREMUM_GRAD_FUSION_PASS
#define FE_OPTIMIZER_FUSION_FULL_EXTREMUM_GRAD_FUSION_PASS

#include <map>
#include <string>
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ExtremumGradFusionPass : public PatternFusionBasePass {
 public:
  ExtremumGradFusionPass();
  ~ExtremumGradFusionPass();

  Status Run(ge::ComputeGraph &graph) override;
  Status Run(ge::ComputeGraph &graph,
             OpsKernelInfoStorePtr opsKernelInfoStorePtr) override;
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
 private:
  bool CheckImplyType();
  bool MatchDx(ge::NodePtr nodeSelect,
                      std::map<std::string, ge::NodePtr> &recordMap);

  bool MatchDy(ge::NodePtr nodeSelect,
                      std::map<std::string, ge::NodePtr> &recordMap);

  Status RunOnePatternFusion(ge::ComputeGraph &graph,
                                    ge::NodePtr equalNode);

  Status DoFusion(ge::ComputeGraph &graph,
                         const std::map<std::string, ge::NodePtr> &recordMap,
                         vector<ge::NodePtr> &fusionNodes);

  Status RemoveNode(ge::ComputeGraph &graph,
                           const std::map<std::string, ge::NodePtr> &recordMap,
                           std::string patternName);

  ge::NodePtr
  CreateExtremumGradNode(ge::ComputeGraph &graph, ge::NodePtr nodeEqual,
                         ge::NodePtr selectDxNode, ge::NodePtr selectDyNode,
                         const std::map<std::string, ge::NodePtr> &recordMap);

  Status SetExtreMumGradOpDesc(ge::OpDescPtr equalOpDesc,
                                      ge::OpDescPtr selectOpDesc,
                                      ge::OpDescPtr extreGradOpDesc);

  Status AdjustAnchor(ge::OutDataAnchorPtr dzInputAnchor,
                             ge::NodePtr equalNode, ge::NodePtr extreGradNode,
                             ge::NodePtr outputDxNode,
                             ge::NodePtr outputDyNode);

  Status ReplaceEdgeDst(ge::OutDataAnchorPtr src,
                               ge::InDataAnchorPtr dst,
                               ge::InDataAnchorPtr newDst);

  Status ReplaceEdgeSrc(ge::OutDataAnchorPtr src,
                               ge::OutDataAnchorPtr newSrc,
                               ge::InDataAnchorPtr dst);

  bool CheckAttrMatch(const std::map<string, ge::NodePtr> &recordMap);

  void SetExtemDataDumpAttr(const std::map<string, ge::NodePtr> &recordMap, vector<ge::NodePtr> &fusionNodes);

  bool CheckEqualOp(ge::NodePtr nodeEqual);

  bool CheckNameScope(const string &nameA, const string &nameB);

  bool CheckZeroConstantOp(ge::NodePtr nodeZeros);

  bool CheckSelectOp(ge::NodePtr nodeSelect, ge::NodePtr nodeEqual);

  bool CheckSameZeroNode(ge::NodePtr nodeZeros,
          const map<string, ge::NodePtr> &recordMap);

  bool CheckSumOp(ge::NodePtr nodeSum, ge::NodePtr nodeEqual);

  ge::NodePtr FindNodeInRecordMap(const map<string, ge::NodePtr> &recordMap,
                                                            string key);
  Status RemoveInputEdges(ge::ComputeGraph &graph, ge::NodePtr node);
  Status RemoveOutputEdges(ge::NodePtr node);
  const std::string CONSTANT = "Const";
  const string FUSED_OP_TYPE = "MaximumGrad/MinimumGrad";
};
}  // namespace fe
#endif /*FE_OPTIMIZER_FUSION_FULL_EXTREMUM_GRAD_FUSION_PASS*/
