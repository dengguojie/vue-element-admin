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
 * \file tbe_matmul_elemwise.cpp
 * \brief tbe matmul and element-wise ops fusion pattern
 */
#include "tbe_matmul_elemwise.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
namespace {
static const string PATTERN_MATMUL = "matmul";
static const string PATTERN_ELTWISE = "eltwise1";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const string PATTERN_OUTPUT = "output";
const int NODE_OUTPUT_SIZE = 1;
}  // namespace

/*
 * @brief:  define Matmul and element-wise op fusion pattern
 *
 *   Matmul + ElemWise
 *
 * fusion node:  Matmul, ElemWise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeMatmulElemwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "TbeMatmulElemwiseFusion";

  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  string passName1 = "TbeMatmulElemwiseFusion1";

  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MATMUL, {OP_PATTERN_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_MATMUL})
      .SetOutputs(PATTERN_MATMUL, {PATTERN_ELTWISE, PATTERN_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName1.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeMatmulElemwiseFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                   vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeMatmulElemwiseFusion!");

  for (auto& item : mapping) {
    auto elemOpdesc = find(item.first->types.begin(), item.first->types.end(), OP_PATTERN_ELEMWISE);
    if (elemOpdesc == item.first->types.end()) {
      continue;
    }
    for (auto& node : item.second) {
      if (node->GetOpDesc()->GetAllOutputsDesc().size() != NODE_OUTPUT_SIZE) {
        OP_LOGI(FUSED_OP_TYPE.c_str(),
                "The number of node[%s] output is [%zu],"
                "which is not equal to one, no need to do fusion.",
                node->GetName().c_str(), node->GetOpDesc()->GetAllOutputsDesc().size());
        return SUCCESS;
      }
    }
  }
  fusionNodes = GetMatchedNodes(mapping);
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(), node);
        fusionNodes.erase(nodePtr);
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeMatmulElemwiseFusion!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("ATbeMatmulElemwiseFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeMatmulElemwiseFusionPass);
}  // namespace fe
