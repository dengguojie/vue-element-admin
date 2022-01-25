/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include <string>
#include <vector>

#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "batch_matmul_v2_dequant_mul_add.h"

namespace fe {
using std::vector;
static const char kPatternMatmul[] = "matmul";
static const char kPatternDequant[] = "dequant";
static const char kPatternEltwise1[] = "eltwise1";
static const char kPatternEltwise2[] = "eltwise2";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";
static const string kFusedOpType = "FusedOp";

/*
 * @brief:  define convolution and single input op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    BatchMatmul + AcendDeQuant + Mul + Add -> BatchMatmul
 *
 * fusion node: BatchMatmul, AcendDeQuant, Mul, Add
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> BatchMatmulV2DequantMulAddFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string pass_name1 = "TbeBatchMatmulV2DequantMulAddFusionPass";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name1);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(kFusedOpType.c_str(), "new an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  // define pattern rules Matmul-->AcendDeQuant
  pattern->AddOpDesc(kPatternMatmul, {OP_PATTERN_BATCH_MATMUL}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternDequant, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternEltwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternEltwise2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({kPatternMatmul})
          .SetOutputs(kPatternMatmul, {kPatternDequant})
          .SetOutputs(kPatternOtherInput, {kPatternDequant})
          .SetOutputs(kPatternDequant, {kPatternEltwise1})
          .SetOutputs(kPatternOtherInput1, {kPatternEltwise1})
          .SetOutputs(kPatternEltwise1, {kPatternEltwise2})
          .SetOutputs(kPatternOtherInput2, {kPatternEltwise2})
          .SetOutputs(kPatternEltwise2, {}, TBE_OUTPUT_BRANCH_SINGLE, true);

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status BatchMatmulV2DequantMulAddFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                            vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do BatchMatmulV2DequantMulAddFusionPass!");

  vector<ge::NodePtr> batchMatmulNodes = GetMatchedNodesByDescName(kPatternMatmul, mapping);
  vector<ge::NodePtr> elemWiseNodes1 = GetMatchedNodesByDescName(kPatternEltwise1, mapping);
  vector<ge::NodePtr> elemWiseNodes2 = GetMatchedNodesByDescName(kPatternEltwise2, mapping);

  bool isElemWiseValid = ((elemWiseNodes1.empty() || elemWiseNodes2.empty() || batchMatmulNodes.empty())
                         || (elemWiseNodes1[0]->GetType() == "Mul" && elemWiseNodes2[0]->GetType() == "Add"));
  if (!isElemWiseValid) {
    OP_LOGD(kFusedOpType.c_str(), "Only support elemwise1 = Mul and elemwise2 = Add");
    return SUCCESS;
  }

  FUSION_PASS_CHECK(elemWiseNodes1[0]->GetAllInDataAnchors().size() != 2,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Mul input nodes must be two"), return SUCCESS);
  FUSION_PASS_CHECK(elemWiseNodes1[0]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Mul output nodes must be one"), return SUCCESS);
  FUSION_PASS_CHECK(elemWiseNodes2[0]->GetAllInDataAnchors().size() != 2,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add input nodes must be two"), return SUCCESS);
  FUSION_PASS_CHECK(elemWiseNodes2[0]->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "Add output nodes must be one"), return SUCCESS);

  fusion_nodes = GetMatchedNodes(mapping);
  // the output_data can't be fused
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc == item.first->types.end()) {
      continue;
    }
    for (auto &node : item.second) {
      auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
      if (node_ptr != fusion_nodes.end()) {
        fusion_nodes.erase(node_ptr);
      }
    }
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do BatchMatmulV2DequantMulAddFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("BatchMatmulV2DequantMulAddFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            BatchMatmulV2DequantMulAddFusionPass);
}  // namespace fe