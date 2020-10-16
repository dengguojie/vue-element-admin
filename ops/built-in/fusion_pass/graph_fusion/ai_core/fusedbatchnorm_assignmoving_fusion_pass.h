
/**
 * @file fusedbatchnorm_assignmoving_fusion_pass.h
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief 
 *
 * @author z00522339
 */

#ifndef FE_FUSEDBATCHNORM_ASSIGNMOVING_FUSION_PASS_H
#define FE_FUSEDBATCHNORM_ASSIGNMOVING_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

using namespace std;

namespace fe {
class FusedBatchNormAssignMovingFusionPass: public PatternFusionBasePass {
 public:
  ~FusedBatchNormAssignMovingFusionPass();

 protected:

  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
  Status Run(ge::ComputeGraph& graph) override;


 private:

  bool IsFusedBatchNormMatched(ge::NodePtr fusedBatchNormnode,Mapping& mapping);
  bool IsAssignMovingMatched(ge::NodePtr node, vector<shared_ptr<OpDesc>> pattern, Mapping &mapping);

  bool MatchAll(ge::ComputeGraph& graph, Mappings& mappings);

  bool Init();
  std::string GetNodeType(ge::NodePtr node);

  /**
   * judege type if exist in types
   */
  bool IsOpTypeExist(const string& type, const vector<string>& types);

  /**
   * judge op_desc whether matched
   */
  bool IsMatched(shared_ptr<OpDesc> op_desc, const ge::NodePtr node, const Mapping& mapping);

  /**
   * print matched result
   */
  void DumpMappings(const Mappings& mappings);

  Status ParseParaFromConst(ge::NodePtr node,float & param ,int index);

  FusedBatchNormAssignMovingFusionPass& AddOpDesc(const string &id, const initializer_list<string> &types,vector<shared_ptr<OpDesc>> &ops);


 private:

  std::vector<shared_ptr<OpDesc>> patternAssignMean;
  std::vector<shared_ptr<OpDesc>> patternAssignVar;
  OpDesc keyOpDesc;

  const string PATTERN_MOVINGMEAN_SUB = "moving_mean_sub";
  const string PATTERN_MOVINGMEAN_MUL = "moving_mean_mul";
  const string PATTERN_MOVINGMEAN_ASSIGNSUB = "moving_mean_assignsub";

  const string PATTERN_MOVINGVAR_SUB = "moving_var_sub";
  const string PATTERN_MOVINGVAR_MUL = "moving_var_mul";
  const string PATTERN_MOVINGVAR_ASSIGNSUB = "moving_var_assignsub";

  const string PATTERN_FUSEDBATCHNORM = "fused_batchnorm";

  const std::string FUSED_BATCHNORM_FACTOR = "fused_batchnorm_factor";

  const string FUSED_OP_TYPE = "FusedBatchNorm";

  /**
   * mark if has error
   */
  bool has_error_ = false;


};

}  // namespace fe

#endif  // FE_FUSEDBATCHNORM_ASSIGNMOVING_FUSION_PASS_H
