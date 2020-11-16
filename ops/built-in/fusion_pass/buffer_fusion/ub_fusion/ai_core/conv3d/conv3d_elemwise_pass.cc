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
 * \file conv3d_elemwise_pass.cpp
 * \brief tbe conv3d + elemwise ops fusion pattern
 */
#include "conv3d_elemwise_pass.h"
#include <string>
#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_CONV3D[] = "conv3d";
static const char PATTERN_ELEM[] = "elemwise";
static const char kPatternOtherInput[] = "otherInput";
static const int DIMS_SIZE = 6;

/*
 * @brief:  define conv3d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv3d --> elemwise
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv3dElemwisePass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string pass_name = "TbeConv3dElemwisePass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);

  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_CONV3D, {OP_PATTERN_CONV3D})
      .AddOpDesc(PATTERN_ELEM, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_CONV3D})
      .SetOutputs(PATTERN_CONV3D, {PATTERN_ELEM})
      .SetOutputs(kPatternOtherInput, {PATTERN_ELEM});
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] mapping: nodes matched by pattern
 * @param [out] fusion_nodes: the nodes of fusion
 * @return uint32_t: fusion status or not.
 */
Status TbeConv3dElemwisePass::GetFusionNodes(const BufferFusionMapping& mapping,
                                             vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do conv3d_elemwise!");

  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PATTERN_ELEM, mapping);
  FUSION_PASS_CHECK(elemNode.empty(),
		    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"),
		    return SUCCESS);

  auto inputs = elemNode[0]->GetOpDesc()->GetAllInputsDesc();
  FUSION_PASS_CHECK(inputs.size() != 2,
		    OP_LOGW(FUSED_OP_TYPE.c_str(), "ElemWise node not match!"),
		    return SUCCESS);

  auto dims0 = elemNode[0]->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  auto dims1 = elemNode[0]->GetOpDesc()->GetInputDesc(1).GetShape().GetDims();

  FUSION_PASS_CHECK(dims0.size() != dims1.size(),
		    OP_LOGW(FUSED_OP_TYPE.c_str(),
			    "the dim sizes of two inputs not equal!"),
		    return SUCCESS);
  FUSION_PASS_CHECK(dims0.size() != DIMS_SIZE,
		    OP_LOGW(FUSED_OP_TYPE.c_str(),
			    "the dim sizes is not 6!"),
		    return SUCCESS);

  auto fusionShape0 = std::vector<int64_t>{dims0[0] * dims0[1], dims0[2], dims0[3] * dims0[4], dims0[5]};
  auto fusionShape1 = std::vector<int64_t>{dims1[0] * dims1[1], dims1[2], dims1[3] * dims1[4], dims1[5]};
  for (size_t i = 0; i < fusionShape0.size(); ++i) {
    if (fusionShape0[i] != fusionShape1[i] && fusionShape0[i] != 1 && fusionShape1[i] != 1) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "the shape can not support to fuse!");
      return SUCCESS;
    }
  }

  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do conv3d_elemwise!");
  
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv3dElemwisePass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeConv3dElemwisePass);
}  // namespace fe
