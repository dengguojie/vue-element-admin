/**
 * @file tbe_conv_bnreduce_fusion_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe convolution and BNReduce ops fusion pattern
 *
 * @version 1.0
 *
 */

#include <string>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "conv2d_quant_stridewrite_pass.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

namespace {
static const string PATTERN_CONV = "convolution";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_STRIDED_WRITE = "stridedwrite";
}

/*
 * @brief:  define convolution and quant and stridewrite op fusion pattern
 *
 *   Convolution-->quant-->stridewrite
 *
 * fusion node: Convolution, quant, stridewrite
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2dQuantStridewriteFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string passName = "TbeConv2dQuantStridewriteFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());

  // define pattern rules
  pattern
      ->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_STRIDED_WRITE, {OP_PATTERN_STRIDED_WRITE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_CONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_QUANT})
      .SetOutputs(PATTERN_QUANT, {PATTERN_STRIDED_WRITE});
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
Status TbeConv2dQuantStridewriteFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                           vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeConv2dQuantStridewriteFusionPass!");
  fusionNodes= GetMatchedNodes(mapping);
    // the outputData can't be fused
    for (auto &item : mapping) {
      auto opdesc = find(item.first->types.begin(),
                         item.first->types.end(),
                         TBE_PATTERN_OUTPUT_NODE);

      if (opdesc != item.first->types.end()) {
        for (auto &node : item.second) {
          auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(),
                              node);
          fusionNodes.erase(nodePtr);
        }
      }
    }
    for (auto &item : mapping) {
      // judge AscendQuant node
      if (item.first->desc_name == PATTERN_QUANT) {
        for (auto &node : item.second) {
          if (node->GetType() == "AscendQuant") {
            OP_LOGD(FUSED_OP_TYPE.c_str(), "AscendQuant is vaild, support ub fusion.");
          } else {
            fusionNodes.clear();
            OP_LOGW(FUSED_OP_TYPE.c_str(), "we only support op type : AscendQuant, "
                    "not support %s.",
                    node->GetType().c_str());
            return SUCCESS;
          }
        }
      }
      // judge StridedWrite node
      if (item.first->desc_name == PATTERN_STRIDED_WRITE) {
        for (auto &node : item.second) {
          if (node->GetType() == "StridedWrite") {
            OP_LOGD(FUSED_OP_TYPE.c_str(), "StridedWrite is vaild, support ub fusion.");
          } else {
            fusionNodes.clear();
            OP_LOGW(FUSED_OP_TYPE.c_str(), "we only support op type : StridedWrite, "
                    "not support %s.",
                    node->GetType().c_str());
            return SUCCESS;
          }
        }
      }
    }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeConv2dQuantStridewriteFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2dQuantStridewriteFusionPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2dQuantStridewriteFusionPass);
}  // namespace fe
