/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(depthwise_conv2d_backprop_filter --> depthwise_conv2d_backprop_filter_d)
 *
 * @author l00463998
 */

#include "depthwise_conv2d_backprop_filter_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const char* DEPTHWISECONV2DBACKPROPFILTER = "DepthwiseConv2DBackpropFilter";
static const std::string PATTERN_DEPTHWISECONV2DBACKPROPFILTER = "DepthwiseConv2DBackpropFilter";

vector<FusionPattern*> DepthwiseConv2DBackpropFilterFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern* pattern = new(std::nothrow) FusionPattern("DepthwiseConv2DBackpropFilterFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_DEPTHWISECONV2DBACKPROPFILTER, {DEPTHWISECONV2DBACKPROPFILTER})
            .SetOutput(PATTERN_DEPTHWISECONV2DBACKPROPFILTER);

  patterns.push_back(pattern);

  return patterns;
}

Status DepthwiseConv2DBackpropFilterFusionPass::Fusion(
    ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes)
{
  std::string fusionOpType = "DepthwiseConv2DBackpropFilterD";
  std::map<int16_t , std::string> depthwiseConv2DBackpropFilterAttrInfo;
  depthwiseConv2DBackpropFilterAttrInfo[1] = "filter_size";
  PatternFusionUtil patternFusionUtil;
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_DEPTHWISECONV2DBACKPROPFILTER, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
   return PARAM_INVALID);
  Status ret = patternFusionUtil.ConstToAttr(graph, fusedNode, fusionOpType,
   depthwiseConv2DBackpropFilterAttrInfo);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "DepthwiseConv2DBackpropFilter has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}
REGISTER_PASS("DepthwiseConv2DBackpropFilterFusionPass",
              BUILT_IN_GRAPH_PASS,
              DepthwiseConv2DBackpropFilterFusionPass);
}
