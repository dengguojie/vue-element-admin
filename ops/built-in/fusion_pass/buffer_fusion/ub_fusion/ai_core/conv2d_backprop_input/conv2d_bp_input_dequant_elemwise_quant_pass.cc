/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*
 * \file conv2d_bp_input_dequant_elemwise_quant_pass.cpp
 * \brief tbe conv2d_backprop_input dequant + elemwise + quant ops fusion pattern
 */
#include "conv2d_bp_input_dequant_elemwise_quant_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_DX[] = "conv2dbackpropinput";
static const char PATTERN_ELEM[] = "elemwise";
static const char PATTERN_DEQUANT[] = "dequant";
static const char PATTERN_QUANT[] = "quant";
static const char PATTERN_OTHER_INPUT[] = "InputData";
static const char PATTERN_OTHER_INPUT1[] = "InputData1";
static const char PATTERN_OTHER_OUTPUT[] = "OutputData";
static vector<string> typelist = {"LeakyRelu", "Prelu"};
/*
 * @brief:  define conv2dbackpropinput op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d_backprop_input --> dequant --> elemwise --> quant
 *                                                   --> output
 *    elemwise only supported leckyrelu/prelu
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeDxDeqElemQuantPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeDxDequantElemwiseQuantFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);

  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  string pass_name0 = "TbeDxDequantElemwiseQuantFusion0";
  BufferFusionPattern* pattern0 = new (std::nothrow) BufferFusionPattern(pass_name0);

  FUSION_PASS_CHECK((pattern0 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name0.c_str());
  pattern0->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT});
  patterns.push_back(pattern0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name0.c_str());

  string pass_name1 = "TbeDxDequantElemwiseQuantFusionDoubleOut";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pass_name1);

  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name1.c_str());
  pattern1->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_OUTPUT, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM})
      .SetOutputs(PATTERN_ELEM, {PATTERN_QUANT, PATTERN_OTHER_OUTPUT}, TBE_OUTPUT_BRANCH_MULTI);
  patterns.push_back(pattern1);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name1.c_str());

  string pass_name2 = "TbeDxDequantElemwise";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pass_name2);

  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name2.c_str());
  pattern2->AddOpDesc(PATTERN_DX, {OP_PATTERN_CONV_BACKPROP_INPUT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_DX})
      .SetOutputs(PATTERN_DX, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEM})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEM});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name2.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeDxDeqElemQuantPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do conv2d_bp_input_elemwise!");

  fusion_nodes = GetMatchedNodes(mapping);

  // buffer fusion do not support dynamic shape now
  vector<ge::NodePtr> dxNodes = GetMatchedNodesByDescName(PATTERN_DX, mapping);
  for (const auto& dxNode : dxNodes){
    vector<int64_t> input0Dims = dxNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
    vector<int64_t> input1Dims = dxNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
    vector<int64_t> allDims;
    allDims.resize(input0Dims.size() + input1Dims.size());
    merge(input0Dims.begin(), input0Dims.end(), input1Dims.begin(), input1Dims.end(), allDims.begin());
    for (auto singleDim : allDims) {
      if (singleDim < 0) {
        fusion_nodes.clear();
        OP_LOGW(FUSED_OP_TYPE.c_str(), "ub fusion not support dynamic shape");
        return SUCCESS;
      }
    }
  }

  // the outputData can't be fused
  for (auto& item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto& node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        fusion_nodes.erase(node_ptr);
      }
    }
  }

  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  FUSION_PASS_CHECK(elemNode.empty(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "get elemNode failed."),
                    return FAILED);
  bool checkElemwise = (elemNode[0]->GetType() == "LeakyRelu" || elemNode[0]->GetType() == "PRelu");
  if (!checkElemwise) {
    fusion_nodes.clear();
    OP_LOGW(FUSED_OP_TYPE.c_str(), "only support LeakyRelu or Prelu");
    return SUCCESS;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv2d_bp_input_dequant_elemwise_quant!");

  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeDxDeqElemQuantPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeDxDeqElemQuantPass);
}  // namespace fe
