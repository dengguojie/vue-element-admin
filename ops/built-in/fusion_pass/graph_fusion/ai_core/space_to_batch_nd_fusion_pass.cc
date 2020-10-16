#include "space_to_batch_nd_fusion_pass.h"
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
static const char *FUSED_NODE = "SpaceToBatchND";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSpaceToBatchND";

vector<FusionPattern*> ConstToAttrSpaceToBatchNdPass::DefinePatterns() {
  vector < FusionPattern * > patterns;
  FusionPattern* pattern = new(std::nothrow) FusionPattern("ConstToAttrSpaceToBatchNdFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
       return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status ConstToAttrSpaceToBatchNdPass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping,
                                             vector<ge::NodePtr> &fusionNodes)
{

  // get fused node
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."), return PARAM_INVALID);

  // build attr infos
  std::string fusionOpType = "SpaceToBatchNDD";
  std::vector<PassAttrInfo> attrInfos;
  PassAttrInfo block_shape = {1, "block_shape", "SetListInt"};
  attrInfos.push_back(block_shape);
  PassAttrInfo paddings = {2, "paddings", "SetListInt"};
  attrInfos.push_back(paddings);

  // build a fusion node op desc
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
           return PARAM_INVALID);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op SpaceToBatchNd Not Supported."),
           return NOT_CHANGED);

  ge::GeTensorDesc first_input_tensor = fusedNode->GetOpDesc()->GetInputDesc(0);
  if (first_input_tensor.GetFormat() != ge::FORMAT_NHWC) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has input which format is not FORMAT_NHWC, graph not changed.");
    return NOT_CHANGED;
  }
  size_t first_dim_num = first_input_tensor.GetShape().GetDimNum();
  if (first_dim_num != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has first input which size is not 4, graph not changed.");
    return NOT_CHANGED;
  }

  ge::GeTensorDesc second_input_tensor = fusedNode->GetOpDesc()->GetInputDesc(1);
  size_t second_dim_num = second_input_tensor.GetShape().GetDimNum();
  int64_t second_dim_0_size = second_input_tensor.GetShape().GetDim(0);
  if (second_dim_num != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has second input which size is not 1, graph not changed.");
    return NOT_CHANGED;
  }
  if (second_dim_0_size != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has second input which size of zero dim is not 2, graph not changed.");
    return NOT_CHANGED;
  }

  // const to attr
  ge::NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, attrInfos, fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusionNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchNDD fusion SUCCESSS!!!!!");
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrSpaceToBatchNdFusion", BUILT_IN_GRAPH_PASS, ConstToAttrSpaceToBatchNdPass);
}
