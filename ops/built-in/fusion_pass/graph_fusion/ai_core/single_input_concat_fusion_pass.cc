/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief
 *
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "single_input_concat_fusion_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
using namespace std;

namespace fe {

static const string CONCAT = "ConcatD";
static const string PATTERN_CONCAT = "concat";

vector<FusionPattern *> SingleInputConcatPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern =
          new (std::nothrow) FusionPattern("SingleInputConcatPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);
  pattern->AddOpDesc(PATTERN_CONCAT, {CONCAT}).SetOutput(PATTERN_CONCAT);

  patterns.push_back(pattern);

  return patterns;
}

static bool ValidConcat(const ge::NodePtr& concatNode) {
  return concatNode->GetInAllNodes().size() <= 1;
}

Status SingleInputConcatPass::Fusion(ge::ComputeGraph &graph,
                                     Mapping &mapping,
                                     vector<ge::NodePtr> &newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "start proc SingleInputConcatPass");

  /* only proc single input concat: delete this op */

  ge::NodePtr concatNode = GetNodeFromMapping(PATTERN_CONCAT, mapping);
  FUSION_PASS_CHECK(concatNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "concatNode is null, fusion failed."), return PARAM_INVALID);

  // do check
  if (!ValidConcat(concatNode))
    return NOT_CHANGED;

  Status ret = graph.RemoveNode(concatNode);
  if (ret != SUCCESS)
    return FAILED;

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SingleInputConcatPass success");
  return SUCCESS;
}
REGISTER_PASS("SingleInputConcatPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, SingleInputConcatPass);
}
