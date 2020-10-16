/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief diag fusion pass(Diag --> DiagD)
 *
 * @author m00516592
 */

#include "diag_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "fp16_t.hpp"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
    // TF opname
  static const float FLOAT_NUM_ZERO = 0;
  static const int32_t INT_NUM_ZERO = 0;
  static const uint16_t UINT_NUM_ZERO = 0;
  static const string PATTERN_DIAG = "Diag";
  static const std::string CONSTANTOP = "Constant";
  static const char *DIAG = "Diag";

template<typename Dtype>
Status AssitHelp(const int32_t n, Dtype& output1){
  Dtype* output = &output1;
  for(int32_t i = 0; i < n; ++i){
    output[(1+n)*i] = 1;
  }
  return SUCCESS;
}

template<typename Dtype>
Status AssitHelpFP16(const int32_t n, Dtype& output1){
  Dtype* output = &output1;
  fp16_t t;
  t.val = 1;
  int32_t xx = 1;
  t = xx;
  for(int32_t i = 0; i < n; ++i){
    output[(1+n)*i] = t.val;
  }
  return SUCCESS;
}

vector<FusionPattern*> DiagFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // diag->diag_d
  // define DiagFusion
  FusionPattern *pattern = new (std::nothrow) FusionPattern("DiagFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_DIAG, {DIAG})
    .SetOutput(PATTERN_DIAG);

  patterns.push_back(pattern);

  return patterns;
}

Status DiagFusionPass::Fusion(ge::ComputeGraph &graph,
                              Mapping &mapping,
                              vector<ge::NodePtr> &fusionNodes)
{
  // diag node
  ge::NodePtr diagVNode = GetNodeFromMapping(PATTERN_DIAG, mapping);
  FUSION_PASS_CHECK(diagVNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "diagVNode is null, fusion failed."), return PARAM_INVALID);

  std::vector<PassAttrInfo> attrInfos;
  const std::string fusionOpType = "DiagD";
  ge::OpDescPtr fusionDescPtr =
        PatternFusionUtil::GetFusionOpDesc(diagVNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."), return PARAM_INVALID);

  // input of diag
  ge::OpDescPtr diagDesc = diagVNode->GetOpDesc();
  FUSION_PASS_CHECK(diagDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "diagVNode's OpDesc is null, fusion failed."),
           return PARAM_INVALID);

  // 寻找diga节点的输入的父节�?
  ge::InDataAnchorPtr diagAnchorPtr0 = diagVNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr constAnchorPtr0 = diagAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();

  // 获取进入diag节点的输入描述：区分const与variable
  ge::GeTensorDesc diagInputTensor = constNode0->GetOpDesc()->GetOutputDesc(0);

  // 获取shape信息
  ge::GeShape diagInputShape = diagInputTensor.GetShape();

  // 获取数据类型
  DataType dataType = diagInputTensor.GetDataType();

  // 获取维度累乘
  int64_t dimNums = diagInputShape.GetShapeSize();

  // GESHAPE->vector
  vector<int64_t> dimInfo = diagInputShape.GetDims();

  // Format
  Format  assitMatrixFormat = diagInputTensor.GetFormat();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc;
  if(dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums*dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);

    Status ret = NnSet(dimNums*dimNums, FLOAT_NUM_ZERO,
                       *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // 定义辅助矩阵输入shape
    vector<int64_t> assitDimInfo;
    for(size_t i = 0;i<2;++i) {
      for(size_t j = 0; j<dimInfo.size(); ++j) {
        assitDimInfo.push_back(dimInfo[j]);
      }
    }

    ge::GeShape assitShape(assitDimInfo);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);

    FUSION_PASS_MAKE_SHARED((assitPtr =
                    std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                    dimNums*dimNums * sizeof(float))), assitPtr = nullptr;
                    return PARAM_INVALID);
  }
  else if(dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums*dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);

    Status ret = NnSet(dimNums*dimNums, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // 定义辅助矩阵输入shape
    vector<int64_t> assitDimInfo;
    for(size_t i = 0;i<2;++i) {
      for(size_t j = 0; j<dimInfo.size(); ++j) {
         assitDimInfo.push_back(dimInfo[j]);
      }
    }

    ge::GeShape assitShape(assitDimInfo);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);

    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                    dimNums*dimNums * sizeof(int32_t))), assitPtr = nullptr;
                    return PARAM_INVALID);
  }
  else if(dataType == ge::DT_FLOAT16) {

    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums*dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"), return PARAM_INVALID);

    Status ret = NnSet(dimNums*dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelpFP16(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelpFP16 failed."), return ret);

    // 定义辅助矩阵输入shape
    vector<int64_t> assitDimInfo;
    for(size_t i = 0;i<2;++i){
      for(size_t j = 0; j<dimInfo.size(); ++j){
        assitDimInfo.push_back(dimInfo[j]);
      }
    }

    ge::GeShape assitShape(assitDimInfo);
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);

    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                    dimNums*dimNums * sizeof(uint16_t))), assitPtr = nullptr; return PARAM_INVALID);
  }

  fusionDescPtr->AddInputDesc(tensorDesc);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
           return NOT_CHANGED);

  vector< ge::GeTensorPtr > weights = { assitPtr };
  ge::OpDescUtils::SetWeights(diagVNode,weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(diagVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  diagDesc->SetType(fusionOpType);

  return SUCCESS;
}
REGISTER_PASS("DiagFusionPass", BUILT_IN_GRAPH_PASS, DiagFusionPass);
}
