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

/*!
 * \file conv2d_dequant_add_mul_quant_pass.cpp
 * \brief tbe conv2d + add + mul + quant ops fusion pattern
 */

#include "conv_clipByValue_fusion_pass.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include <string>
#include <vector>

namespace fe {
using std::vector;

static const string PATTERN_CONV = "Convolution";
static const string PATTERN_ELEMWISE = "elemwise";
static const string PATTERN_OTHER_INPUT1 = "otherInput1";
static const string PATTERN_OTHER_INPUT2 = "otherInput2";
/*
  * @brief: define convlution and clipByValue op fusion pattern
  * 
  * pattern configuration limit:
  * 1. total min value must be 1 for all head candidated desc.
  * 2. any head candidated desc max value must be 1.
  * 3. output desc can not be itself
  * 
  *           Convolution --> ClipByValue
  *           ClipByValue --> Convolution
  * 
  *  fusion node: Convolution, ClipByValue
  * 
  *  @ return BufferFusionPatternL return all valid patterns.
  */
 vector<BufferFusionPattern *> ConvClipByValueFusionPass::DefinePatterns() {
   vector<BufferFusionPattern *> patterns;

   string passName = "ConvClipByValueFusionPass";
   BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
   FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE("new an object failed."), return patterns);
   OP_LOGD("Start to define %s pass pattern.", passName.c_str());
   /* define pattern        Convolution  -->  Elemwise
    *                                        /      |
    *                                     input1  input2
    */
   pattern 
       ->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .SetHead({PATTERN_CONV})
       .SetOutputs(PATTERN_CONV, {PATTERN_ELEMWISE})
       .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_ELEMWISE})
       .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_ELEMWISE});

   patterns.push_back(pattern);
   OP_LOGD("End of defination of %s pass pattern.", passName.c_str());


   string passName1 = "ClipByValueConvFusionPass";
   BufferFusionPattern *pattern1 = new (std::nothrow) BufferFusionPattern(passName1);
   FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE("new an object failed."), return patterns);
   OP_LOGD("Start to define %s pass pattern.", passName1.c_str());
   /* defube oatterb        Elemwise  -->  Convolution
    *                                       /      |
    *                                    input1  input2
    */
   pattern1
       ->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .AddOpDesc(PATTERN_OTHER_INPUT2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
       .SetHead({PATTERN_ELEMWISE})
       .SetOutputs(PATTERN_ELEMWISE, {PATTERN_CONV})
       .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_CONV})
       .SetOutputs(PATTERN_OTHER_INPUT2, {PATTERN_CONV});

   patterns.push_back(pattern1);
   OP_LOGD("End of defination of %s pass pattern.", passName1.c_str());

   return patterns;
 }

  /*
  * @brief: parse nodes matched in mapping and call DoFusion
  * @param [in] graph: original graph
  * @param [out] mapping: nodes matched by pattern
  * @return bool: fusion status ok or not
  */
  Status ConvClipByValueFusionPass::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    OP_LOGD("Begin of ConvClipByValue ub fusion pass!!");
    vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEMWISE, mapping);

    if (!elemNode.empty()) {
      if (elemNode[0]->GetType() != "ClipByValue") {
        OP_LOGI(
            "The optype of node[%s] should be ClipByValue,"
            "but actually is [%s], no need to do fusion.",
            elemNode[0]->GetName().c_str(), elemNode[0]->GetType().c_str());
          return SUCCESS;
      }
    }

    fusionNodes = GetMatchedNodes(mapping);
    OP_LOGD("End of ConvClipByValue ub fusion pass!");
    return SUCCESS;
  }
  REGISTER_BUFFER_FUSION_PASS("ConvClipByValueFusionPass",
                              BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                              ConvClipByValueFusionPass);
  }  // namespace fe