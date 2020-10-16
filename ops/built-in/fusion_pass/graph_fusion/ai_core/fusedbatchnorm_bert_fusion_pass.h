/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fusedbatchnormgrad_bert fusion pass
 *
 * @author z00445087
 */

#ifndef DAVINCI_FUSEDBATCHNORM_BERT_FUSION_PASS_H
#define DAVINCI_FUSEDBATCHNORM_BERT_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{

class FusedBatchNormBertFusionPass: public PatternFusionBasePass
{
 protected:

  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;

 private:
  vector<ge::NodePtr> GetNodesFromMapping(const string &id, Mapping& mapping);
  const string FUSED_OP_TYPE = "BNTrainingReduce_BNTrainingUpdateV2";
};

}  // namespace domi
#endif //DAVINCI_FUSEDBATCHNORM_BERT_FUSION_PASS_H
