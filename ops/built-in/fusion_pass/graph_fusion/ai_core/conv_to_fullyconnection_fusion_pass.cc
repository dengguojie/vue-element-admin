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
 * \file conv_to_fullyconnection_fusion_pass.cpp
 * \brief fuse conv to fullyconnection
 */
#include "conv_to_fullyconnection_fusion_pass.h"
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "conv_fusion_pass_base.h"
#include "error_util.h"

namespace fe {
static const char PATTERN_CONV[] = "conv";
static const string FC = "FullyConnection";
static const string SIGMOID = "Sigmoid";
static const string DEQUANT = "AscendDequant";
static const string QUANT = "AscendQuant";
static const string REQUANT = "AscendRequant";
static const char ATTR_GROUPS[] = "groups";
static const char ATTR_PADS[] = "pads";
static const char ATTR_NUM_OUTPUT[] = "num_output";
static const std::map<ge::Format, std::map<std::string, int32_t>> FE_AXIS_INDEX_OF_FORMAT = {
    {ge::FORMAT_NCHW, {{"N", NCHW_DIM_N}, {"C", NCHW_DIM_C}, {"H", NCHW_DIM_H}, {"W", NCHW_DIM_W}}},
    {ge::FORMAT_HWCN, {{"N", HWCN_DIM_N}, {"C", HWCN_DIM_C}, {"H", HWCN_DIM_H}, {"W", HWCN_DIM_W}}},
    {ge::FORMAT_NHWC, {{"N", NHWC_DIM_N}, {"C", NHWC_DIM_C}, {"H", NHWC_DIM_H}, {"W", NHWC_DIM_W}}}};

vector<FusionPattern*> ConvToFullyConnectionFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConvToFullyConnectionFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object not success."), return patterns);

  pattern->AddOpDesc(PATTERN_CONV, {CONV2D}).SetOutput(PATTERN_CONV);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern ConvToFullyConnectionFusionPass success.");
  return patterns;
}

int64_t ConvToFullyConnectionFusionPass::GetDimByAxisName(const ge::GeTensorDesc& tensor, const string& axis) {
  ge::Format format = tensor.GetFormat();
  int32_t index = 0;
  auto iter = FE_AXIS_INDEX_OF_FORMAT.find(format);
  if (iter != FE_AXIS_INDEX_OF_FORMAT.end()) {
    auto iter2 = iter->second.find(axis);
    if (iter2 != iter->second.end()) {
      index = iter2->second;
    } else {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Do not support this axis %s", axis.c_str());
      index = -1;
    }
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Do not support this format %s",
            ge::TypeUtils::FormatToSerialString(format).c_str());
    index = -1;
  }
  ge::GeShape shape = tensor.GetShape();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "format %s, axis %s, index %d, dim %ld",
          ge::TypeUtils::FormatToSerialString(format).c_str(), axis.c_str(), index, shape.GetDim(index));
  return shape.GetDim(index);
}

int32_t ConvToFullyConnectionFusionPass::GetIndexByAxisName(const ge::GeTensorDesc& tensor, const string& axis) {
  ge::Format format = tensor.GetFormat();
  int32_t index = 0;
  auto iter = FE_AXIS_INDEX_OF_FORMAT.find(format);
  if (iter != FE_AXIS_INDEX_OF_FORMAT.end()) {
    auto iter2 = iter->second.find(axis);
    if (iter2 != iter->second.end()) {
      index = iter2->second;
    } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Format %s does not support this axis %s",
              ge::TypeUtils::FormatToSerialString(format).c_str(), axis.c_str());
      index = -1;
    }
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Do not support this format %s",
            ge::TypeUtils::FormatToSerialString(format).c_str());
    index = -1;
  }
  return index;
}

Status ConvToFullyConnectionFusionPass::CheckHWCEqual(const ge::GeTensorDesc& xTensor,
                                                      const ge::GeTensorDesc& filterTensor) {
  ge::GeShape shapeX = xTensor.GetShape();
  ge::GeShape shapeFilter = filterTensor.GetShape();
  int32_t xAixsIndexH = GetIndexByAxisName(xTensor, "H");
  int32_t filterAixsIndexH = GetIndexByAxisName(filterTensor, "H");
  int32_t xAixsIndexW = GetIndexByAxisName(xTensor, "W");
  int32_t filterAixsIndexW = GetIndexByAxisName(filterTensor, "W");
  int32_t xAixsIndexC = GetIndexByAxisName(xTensor, "C");
  int32_t filterAixsIndexC = GetIndexByAxisName(filterTensor, "C");
  if ((xAixsIndexH == -1) ||
      (filterAixsIndexH == -1) ||
      (xAixsIndexW == -1) ||
      (filterAixsIndexW == -1) ||
      (xAixsIndexC == -1) ||
      (filterAixsIndexC == -1)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ConvToFullyConnectionFusionPass cannot be applied for the format or axis.");
    return NOT_CHANGED;
  }

  int64_t xAixsH = shapeX.GetDim(xAixsIndexH);
  int64_t filterAixsH = shapeFilter.GetDim(filterAixsIndexH);
  int64_t xAixsW = shapeX.GetDim(xAixsIndexW);
  int64_t filterAixsW = shapeFilter.GetDim(filterAixsIndexW);
  int64_t xAixsC = shapeX.GetDim(xAixsIndexC);
  int64_t filterAixsC = shapeFilter.GetDim(filterAixsIndexC);

  if (PatternFusionUtil::IsUnknownShape(xAixsH) ||
      PatternFusionUtil::IsUnknownShape(filterAixsH) ||
      PatternFusionUtil::IsUnknownShape(xAixsW) ||
      PatternFusionUtil::IsUnknownShape(filterAixsW) ||
      PatternFusionUtil::IsUnknownShape(xAixsC) ||
      PatternFusionUtil::IsUnknownShape(filterAixsC)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "ConvToFullyConnectionFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "HWC of x is [%ld, %ld, %ld], filter is [%ld, %ld, %ld].", xAixsH, xAixsW, xAixsC,
          filterAixsH, filterAixsW, filterAixsC);

  bool flag = xAixsH && xAixsW && xAixsC && filterAixsH && filterAixsW && filterAixsC;
  FUSION_PASS_CHECK(!flag, OP_LOGD(FUSED_OP_TYPE.c_str(), "HWC invalid."), return FAILED);

  flag = (xAixsH == filterAixsH) && (xAixsW == filterAixsW) && (xAixsC == filterAixsC);
  FUSION_PASS_CHECK(!flag, OP_LOGD(FUSED_OP_TYPE.c_str(), "HWC of x and filter are not equal."), return FAILED);
  return SUCCESS;
}

Status ConvToFullyConnectionFusionPass::CheckFusionParm(ge::NodePtr convNode) {
  string convNodeName = convNode->GetName();

  ge::GeTensorDesc xInputDesc = convNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc filterInputDesc = convNode->GetOpDesc()->GetInputDesc(1);

  FUSION_PASS_CHECK(convNode->GetOutAllNodes().size() != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Conv out node num should be one."), return FAILED);

  ge::NodePtr convNextNode = convNode->GetOutAllNodes().at(0);
  string nextOpType = convNextNode->GetOpDesc()->GetType();
  FUSION_PASS_CHECK((nextOpType == QUANT) || (nextOpType == REQUANT),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Not support fc+requant or fc+quant ub fusion."), return FAILED);
  for (unsigned i = 0; i < convNextNode->GetOutAllNodes().size(); i++) {
    string nextNextOpType = convNextNode->GetOutAllNodes().at(i)->GetOpDesc()->GetType();
    OP_LOGD(FUSED_OP_TYPE.c_str(), "conv2d next next node is %s", nextNextOpType.c_str());
    FUSION_PASS_CHECK((nextNextOpType == SIGMOID) && (nextOpType == DEQUANT),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Not support effient_net fc+dequant+sigmod ub fusion."),
                      return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "%s, x format %s, ori format %s, filter format %s, ori format %s.",
          convNodeName.c_str(), TypeUtils::FormatToSerialString(xInputDesc.GetFormat()).c_str(),
          TypeUtils::FormatToSerialString(xInputDesc.GetOriginFormat()).c_str(),
          TypeUtils::FormatToSerialString(filterInputDesc.GetFormat()).c_str(),
          TypeUtils::FormatToSerialString(filterInputDesc.GetOriginFormat()).c_str());
  FUSION_PASS_CHECK(CheckHWCEqual(xInputDesc, filterInputDesc) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "HWC of x and filter are not equal."), return FAILED);

  int32_t groups = 0;
  (void)ge::AttrUtils::GetInt(convNode->GetOpDesc(), ATTR_GROUPS, groups);
  FUSION_PASS_CHECK(groups != 1, OP_LOGI(FUSED_OP_TYPE.c_str(), "groups of should be 1, actual is %ld.", groups),
                    return FAILED);

  vector<int64_t> convPads;
  vector<int64_t> convPadsTarget = {0, 0, 0, 0};
  (void)ge::AttrUtils::GetListInt(convNode->GetOpDesc(), ATTR_PADS, convPads);
  FUSION_PASS_CHECK(convPads != convPadsTarget, OP_LOGI(FUSED_OP_TYPE.c_str(), "convPads of should be [0, 0, 0, 0]."),
                    return FAILED);
  return SUCCESS;
}

Status ConvToFullyConnectionFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr convNode = GetNodeFromMapping(PATTERN_CONV, mapping);
  FUSION_PASS_CHECK(convNode == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new convNode not success."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(CheckFusionParm(convNode) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Can not do fusion to node, %s.", convNode->GetName().c_str()),
                    return NOT_CHANGED);

  ge::OpDescPtr convOp = convNode->GetOpDesc();
  FUSION_PASS_CHECK(convOp == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "convOp not success."), return PARAM_INVALID);

  // set optype to fullyconnection
  convOp->SetType(FC);

  // set attr num_output of fullyconnection
  ge::GeTensorDesc outputDesc = convNode->GetOpDesc()->GetOutputDesc(0);
  int32_t outAixsC = GetDimByAxisName(outputDesc, "C");
  OP_LOGD(FUSED_OP_TYPE.c_str(), "outAixsC of conv is %ld.", outAixsC);
  (void)ge::AttrUtils::SetInt(convOp, ATTR_NUM_OUTPUT, outAixsC);

  // >>> start: change bias output format when shape is 1
  auto bias_tensor = convNode->GetOpDesc()->MutableInputDesc("bias");
  if (bias_tensor != nullptr) {
    auto bias_shape = bias_tensor->MutableShape().GetDims();
    auto format = bias_tensor->GetFormat();
    bool valid = format == ge::FORMAT_ND && bias_shape.size() > 0 && bias_shape[0] == 1;
    if (valid) {
      bias_tensor->SetFormat(ge::FORMAT_NCHW);
      bias_tensor->SetOriginFormat(ge::FORMAT_NCHW);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "change bias format from ND to NCHW.");
    }
  }
  // <<< end: change bias output format when shape is 1

  // update tensor name
  std::map<string, uint32_t> inputNameMap = convOp->GetAllInputName();
  std::map<string, uint32_t> inputNameMapNew;
  for (auto inputName : inputNameMap) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "conv %ld th input is %s.", inputName.second, inputName.first.c_str());
    string key = inputName.first;
    uint32_t value = inputName.second;
    if (inputName.first == "filter") {
      key = "w";
    } else if (inputName.first == "bias") {
      key = "b";
    }
    inputNameMapNew.insert(make_pair(key, value));
  }

  for (auto inputName : inputNameMapNew) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "after conv %ld th input is %s.", inputName.second, inputName.first.c_str());
  }
  FUSION_PASS_CHECK(false == convOp->UpdateInputName(inputNameMapNew),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "UpdateInputName conv failed."), return FAILED);

  inputNameMap = convOp->GetAllInputName();
  for (auto inputName : inputNameMap) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "after1 conv %ld th input is %s.", inputName.second, inputName.first.c_str());
  }

  fusionNodes.push_back(convNode);
  return SUCCESS;
}
REGISTER_PASS("ConvToFullyConnectionFusionPass", BUILT_IN_GRAPH_PASS, ConvToFullyConnectionFusionPass);
}  // namespace fe
