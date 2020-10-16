#include "range_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"


using namespace std;
using namespace ge;

namespace fe {
  const int64_t realDimCnt = 1;
  static const int32_t INT_NUM_ZERO = 0;
  static const float FLOAT_NUM_ZERO = 0;
  static const string PATTERN_RANGE = "Range";
  const std::string CONSTANTOP = "Constant";
  const char *RANGE = "Range";

Status assist_float_help(const int32_t n, float* output) {
  for (int32_t i=0; i<n; i++){
  output[i] = float(i);
  }
  return SUCCESS;
}

Status assist_int_help(const int32_t n, int32_t* output) {
  for (int32_t i=0; i<n; i++) {
    output[i] = i;
  }
  return SUCCESS;
}

vector<FusionPattern*> RangeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("RangeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),  return patterns);

  pattern->AddOpDesc(PATTERN_RANGE, {RANGE})
    .SetOutput(PATTERN_RANGE);
  patterns.push_back(pattern);

  return patterns;
}

Status RangeFusionPass::Fusion(ge::ComputeGraph &graph,
                               Mapping &mapping,
                               vector<ge::NodePtr> &fusionNodes)
{
  ge::NodePtr rangeVNode = GetNodeFromMapping(PATTERN_RANGE, mapping);
  FUSION_PASS_CHECK(rangeVNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "rangeVNode is null, fusion failed."), return PARAM_INVALID);

  ge::OpDescPtr rangeDesc = rangeVNode->GetOpDesc();
  FUSION_PASS_CHECK(rangeDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "rangeDesc is null, fusion failed."), return PARAM_INVALID);

  ge::InDataAnchorPtr range_anchor_ptr0 = rangeVNode->GetInDataAnchor(0);
  ge::InDataAnchorPtr range_anchor_ptr1 = rangeVNode->GetInDataAnchor(1);
  ge::InDataAnchorPtr range_anchor_ptr2 = rangeVNode->GetInDataAnchor(2);
  ge::OutDataAnchorPtr constAnchorPtr0 = range_anchor_ptr0->GetPeerOutAnchor();
  ge::OutDataAnchorPtr constAnchorPtr1 = range_anchor_ptr1->GetPeerOutAnchor();
  ge::OutDataAnchorPtr constAnchorPtr2 = range_anchor_ptr2->GetPeerOutAnchor();
  ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();
  ge::NodePtr constNode1 = constAnchorPtr1->GetOwnerNode();
  ge::NodePtr constNode2 = constAnchorPtr2->GetOwnerNode();
  ge::DataType constType0 = constNode0->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType constType1 = constNode1->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType constType2 = constNode2->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::Format constformat0 = constNode0->GetOpDesc()->GetOutputDesc(0).GetFormat();

  std::string fusionOpType="RangeD";
  std::vector<PassAttrInfo> rangeAttrInfo;
  if(constType0==ge::DT_INT32 && constType1==ge::DT_INT32 && constType2==ge::DT_INT32) {
    PassAttrInfo start_attr={0, "start", "SetInt"};
    PassAttrInfo limit_attr={1, "limit", "SetInt"};
    PassAttrInfo delta_attr={2, "delta", "SetInt"};
    rangeAttrInfo.push_back(start_attr);
    rangeAttrInfo.push_back(limit_attr);
    rangeAttrInfo.push_back(delta_attr);
  }else {
    PassAttrInfo start_attr={0, "start", "SetFloat"};
    PassAttrInfo limit_attr={1, "limit", "SetFloat"};
    PassAttrInfo delta_attr={2, "delta", "SetFloat"};
    rangeAttrInfo.push_back(start_attr);
    rangeAttrInfo.push_back(limit_attr);
    rangeAttrInfo.push_back(delta_attr);
  }

  ge::NodePtr fusionNode = nullptr;
  std::string nodeName = rangeVNode->GetName();
  Status ret = PatternFusionUtil::ConstToAttrWithType(graph, rangeVNode, fusionOpType, rangeAttrInfo);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Range has input which is not a constant, graph not changed");
    return NOT_CHANGED;
  }
  for (auto node: graph.GetDirectNode()) {
    if(nodeName == node->GetName()) {
      fusionNode = node;
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Find FusionNode");
      break;
    }
  }
  FUSION_PASS_CHECK(rangeDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "FusionNode is full, fusion failed."), return PARAM_INVALID);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusionNode);

  float res;
  if(constType0==ge::DT_INT32 && constType1==ge::DT_INT32 && constType2==ge::DT_INT32) {
    int constData1;
    int constData2;
    int constData3;
    if (ge::GRAPH_SUCCESS != op.GetAttr("start", constData1)) {
      return GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != op.GetAttr("limit", constData2)) {
      return GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != op.GetAttr("delta", constData3)) {
      return GRAPH_FAILED;
    }
    int assist_num;
    int assist_num_one;
    assist_num = abs(constData2 - constData1);
    assist_num_one = abs(constData3);
    res = ceil(float(assist_num) / assist_num_one);
  }else {
    float constData1;
    float constData2;
    float constData3;
    if (ge::GRAPH_SUCCESS != op.GetAttr("start", constData1)) {
      return GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != op.GetAttr("limit", constData2)) {
      return GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != op.GetAttr("delta", constData3)) {
      return GRAPH_FAILED;
    }
    float assist_num;
    float assist_num_one;
    assist_num = abs(constData2 - constData1);
    assist_num_one = abs(constData3);
    res = ceil(assist_num / assist_num_one);
  }

  std::vector<int64_t> dimsIn;
  dimsIn.push_back(res);
  ge::GeShape assistShape(dimsIn);
  ge::GeTensorDesc rangeInputTensor;
  if(constType0==ge::DT_INT32 && constType1==ge::DT_INT32 && constType2==ge::DT_INT32) {
    rangeInputTensor.SetDataType(ge::DT_INT32);
  }else {
    rangeInputTensor.SetDataType(ge::DT_FLOAT);
  }

  rangeInputTensor.SetFormat(constformat0);
  rangeInputTensor.SetShape(assistShape);
  TensorUtils::SetRealDimCnt(rangeInputTensor, realDimCnt);

  ge::GeShape rangeInputShape = rangeInputTensor.GetShape();
  DataType dataType = rangeInputTensor.GetDataType();
  int64_t dimNums = rangeInputShape.GetShapeSize();

  // GESHAPE->vector
  vector<int64_t> dimInfo = rangeInputShape.GetDims();
  ge::GeTensorPtr assitPtr = nullptr;
  if(dataType == ge::DT_INT32) {
    unique_ptr<int32_t> inputAssit(new (std::nothrow) int32_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);
    ret = assist_int_help(dimNums, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(rangeInputTensor, reinterpret_cast<uint8_t *>(inputAssit.get()), dimNums * sizeof(int32_t))), assitPtr = nullptr; return PARAM_INVALID);
  }else {
    unique_ptr<float> inputAssit(new (std::nothrow) float[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);
    ret = assist_float_help(dimNums, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(rangeInputTensor, reinterpret_cast<uint8_t *>(inputAssit.get()), dimNums * sizeof(float))), assitPtr = nullptr; return PARAM_INVALID);
  }

  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(fusionNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(fusionNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  rangeDesc->SetType("RangeD");
  fusionNodes.push_back(fusionNode);
  return SUCCESS;
}

  REGISTER_PASS("RangeFusionPass", BUILT_IN_GRAPH_PASS, RangeFusionPass);
}
