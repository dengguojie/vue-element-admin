/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief split fusion pass(strided_slice --> strided_slice_d)
 *
 */

#include "strided_slice_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "StridedSlice";

static const std::string PATTERN_FUSEDNODE = "FusedNodeStridedSlice";

vector<FusionPattern*> ConstToAttrStridedSlicePass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern *pattern =
      new(std::nothrow) FusionPattern("ConstToAttrStridedSliceFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrStridedSlicePass::Fusion(ge::ComputeGraph &graph,
                                           Mapping &mapping,
                                           vector<ge::NodePtr> &fusionNodes)
{
  std::string fusion_op_type = "StridedSliceD";
  // PatternFusionUtil patternFusionUtil;
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  ge::OpDescPtr fuseDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fuseDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node's OpDesc is null, fusion failed."),
           return PARAM_INVALID);
  vector<int64_t> dims = fuseDesc->GetOutputDesc(0).GetShape().GetDims();
  for(int64_t ele : dims) {
    if (ele == 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The output of strided slice's dim is 0, need go to aicpu");
      return NOT_CHANGED;
    }
  }

  FUSION_PASS_CHECK(fused_node == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "fused_node is null, fusion failed."),
           return PARAM_INVALID);

  std::vector<PassAttrInfo> attr_infos = {
      {1, "begin", "SetListInt"},
      {2, "end", "SetListInt"},
      {3, "strides", "SetListInt"}
  };

  ge::OpDescPtr desc_ptr = PatternFusionUtil::GetFusionOpDesc(fused_node,
                                                              fusion_op_type,
                                                              attr_infos);
  FUSION_PASS_CHECK(desc_ptr == nullptr,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "The input is not a constant."),
           return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckOpSupported(desc_ptr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
           return NOT_CHANGED);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);

  std::vector<int64_t> strides;
  if (!TbeFusionPassUtil::GetConstIntData(op, "strides", strides)) {
    return FAILED;
  }
  if (!strides.empty() && strides[strides.size()-1] != 1) {
     OP_LOGI(FUSED_OP_TYPE.c_str(), "StridedSlice last value of strides is not 1, graph not changed.");
     return NOT_CHANGED;
   }

  int64_t newmask = 0;
  int64_t shrinkmask = 0;
  int64_t new_axis_flag = 0;
  size_t delete_flag = 0;
  ge::Shape inputShape = op.GetInputDesc("x").GetShape();
  size_t dim_num = inputShape.GetDimNum();
  size_t base_number = 2.0;
  bool shrink_last_dim_flag = false;

  if ((ge::GRAPH_SUCCESS != op.GetAttr("new_axis_mask", newmask)) ||
      (ge::GRAPH_SUCCESS != op.GetAttr("shrink_axis_mask", shrinkmask))) {
    OP_LOGE(FUSED_OP_TYPE.c_str(),
        "op strided_slice get attribute new axis mask or shrink axis mask failed");
    return FAILED;
  }

  for (size_t i = 0; i < dim_num; i++) {
    if ((newmask & ((int64_t)pow(base_number, i))) ==
                      ((int64_t)pow(base_number, i))) {
      new_axis_flag += 1;
    }
    if ((shrinkmask & ((int64_t)pow(base_number, i))) ==
                      ((int64_t)pow(base_number, i))) {
      delete_flag += 1;
      if (i == dim_num - 1) {
        shrink_last_dim_flag = true;
      }
    }
  }

  if (dim_num + new_axis_flag - delete_flag == 1){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The output of strided slice is 1D, need go to aicpu");
  }


  if ((shrink_last_dim_flag) && (dim_num != 1)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Shrink the last dim, need go to aicpu");
    return NOT_CHANGED;
  }

  std::vector<int64_t> begins;
  if (!TbeFusionPassUtil::GetConstIntData(op, "begin", begins)) {
    return FAILED;
  }

  std::vector<int64_t> ends;
  if (!TbeFusionPassUtil::GetConstIntData(op, "end", ends)) {
    return FAILED;
  }
  ge::NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph,
                                                      fused_node,
                                                      fusion_op_type,
                                                      attr_infos,
                                                      fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "StridedSlice has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusionNode);
  return SUCCESS;
}

REGISTER_PASS("ConstToAttrStridedSliceFusion", BUILT_IN_GRAPH_PASS, ConstToAttrStridedSlicePass);
}
