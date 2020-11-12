/**
 * @file tbe_fullyconnection_elemwise_fusion_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe multi-output fusion pattern
 *
 * @version 1.0
 *
 */

#include "tbe_fullyconnection_elemwise_fusion_pass.h"
#include <string>
#include <vector>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const string PATTERN_FC_MATMUL = "FullyConnection/MatMul";     // desc name
static const string PATTERN_DEQUANT = "dequant";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_ELTWISE1 = "eltwise1";      // desc name
static const string PATTERN_ELTWISE2 = "eltwise2";      // desc name
static const string PATTERN_OTHER_INPUT = "InputData";  // desc name
static const vector<string> elemWiseWhiteList = {
  "Elu", "LeakyRelu", "Gelu", "Softsign", "Relu6", "Relu", "Softplus", "Sigmoid", "Tanh", "Selu",
  "GeluGrad", "Add", "AddN", "FastGelu", "FastGeluGrad", "Eltwise", "PRelu", "Mul", "Power", "Relu6D"};

/*
 * @brief:  define fully connection elemwise fusion pattern
 *
 * pattern configuration limit:
 *
 * FullyConnection/MatMUL + (AscendDequant) +（ReLU/LeakyReLU) + (eltwise)
 *
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeFullyconnectionElemwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string passName = "TbeFullyconnectionElemwiseDequantFusionPass";

  BufferFusionPattern *pattern =
          new (std::nothrow) BufferFusionPattern(passName, TBE_FUSION_OP_NUM_MAX);
  if (pattern == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed.");
    return patterns;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_FC_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_ELTWISE2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({PATTERN_FC_MATMUL})
          .SetOutputs(PATTERN_FC_MATMUL, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
          .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELTWISE1})
          .SetOutputs(PATTERN_ELTWISE1, {PATTERN_ELTWISE2})
          .SetOutputs(PATTERN_ELTWISE2, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeFullyconnectionElemwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD("Begin to do TbeFullyconnectionElemwiseFusionPass!");
  fusionNodes = GetMatchedNodes(mapping);
  vector<ge::NodePtr> fcNodes = GetMatchedNodesByDescName(PATTERN_FC_MATMUL, mapping);
  vector<ge::NodePtr> reluNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE1, mapping);
  vector<ge::NodePtr> elemWiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE2, mapping);
  // check whether the fc/matmul op
  for (const auto &fcNode : fcNodes) {
    if (fcNode->GetType() != "FullyConnection" && fcNode->GetType() != "MatMul" && fcNode->GetType() != "MatMulV2") {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              fcNode->GetName().c_str(), fcNode->GetType().c_str());
      return SUCCESS;
    }
  }

  // check whether the relu/leakyrelu op
  if (elemWiseNodes.empty()) {
    for (const auto &reluNode : reluNodes) {
      if (reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu" &&
        find(elemWiseWhiteList.begin(), elemWiseWhiteList.end(), reluNode->GetType()) == elemWiseWhiteList.end()) {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
    }
  } else {
    for (const auto &reluNode : reluNodes) {
      if (reluNode->GetType() != "Relu" && reluNode->GetType() != "LeakyRelu") {
        fusionNodes.clear();
        OP_LOGD(FUSED_OP_TYPE.c_str(),
                "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
                reluNode->GetName().c_str(), reluNode->GetType().c_str());
        return SUCCESS;
      }
    }
  }

  // check whether the EltWise op is in the whitelist
  for (const auto &elemWiseNode : elemWiseNodes) {
    if (find(elemWiseWhiteList.begin(), elemWiseWhiteList.end(), elemWiseNode->GetType()) ==
        elemWiseWhiteList.end()) {
      fusionNodes.clear();
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
              elemWiseNode->GetName().c_str(), elemWiseNode->GetType().c_str());
      return SUCCESS;
    }
  }
  OP_LOGD("End to do TbeFullyconnectionElemwiseFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeFullyconnectionElemwiseDequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeFullyconnectionElemwiseFusionPass);
}  // namespace fe
