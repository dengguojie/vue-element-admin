/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief avg_pool_1d fusion pass(avg_pool_1d --> avg_pool_1dD)
 */

#include "avg_pool_1d_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_AVGPOOL1D = "AvgPool1D";
static const std::string CONSTANTOP = "Constant";
static const char *AVGPOOL1D = "AvgPool1D";

Status AvgPool1DFusionPass::AvgValueTableGen(vector<int64_t> dimInfo, int64_t kernelSize,
                        int64_t strideSize, vector<int64_t> padding,
                        bool ceilMode, bool countIncludePad,
                        ge::Format dataFormat, ge::DataType inputType,
                        vector<int64_t> &assitDimInfo, uint16_t *output) {
  if (output == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "output pointer is null!");
    return FAILED;
  }
  int64_t nInput = 1;
  int64_t cInput = 1;
  int64_t hInput = 1;
  int64_t wInInput = 0;

  int64_t kSize = kernelSize;
  int64_t stride = strideSize;
  int64_t padl = padding[0];
  int64_t padr = padding[1];

  // dimInfo must NCHW
  wInInput = dimInfo[3];

  int64_t nOutput = nInput;
  int64_t cOutput = cInput;
  int64_t hOutput = hInput;
  int64_t wOutput = 0;

  if (ceilMode) {
    wOutput = (wInInput + padl + padr - kSize + stride - 1) / stride + 1;
  } else {
    wOutput = ((wInInput + padl + padr) - kSize) / stride + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and stride is big, but kernel is small, return nan
    if (((wOutput - 1) * stride) >= (wInInput + padl)) {
      wOutput--;
    }
  }
  padr = (wOutput - 1) * stride + kSize - wInInput - padl;
  // set output data
  float dataNum = 0.0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t outOffsetPoint = 0;
  for (int64_t n = 0; n < nOutput; n++) {
    for (int64_t c = 0; c < cOutput; c++) {
      for (int64_t h = 0; h < hOutput; h++) {
        for (int64_t w = 0; w < wOutput; w++) {
          start = stride * w;
          end = stride * w + kSize;
          outOffsetPoint = n * cOutput * hOutput * wOutput +
                           c * hOutput * wOutput + h * wOutput + w;
          if (!countIncludePad) {
            start = max(start, padl);
            end = min(end, wInInput + padl);
          } else {
            end = min(end, wInInput + padl + padr);
          }
          dataNum = end - start;
          fp16_t tmp;
          tmp = 1.0 / dataNum;
          output[outOffsetPoint] = tmp.val;
        }
      }
    }
  }
  if (dataFormat == FORMAT_NHWC) {
    assitDimInfo.push_back(nOutput);
    assitDimInfo.push_back(hOutput);
    assitDimInfo.push_back(wOutput);
    assitDimInfo.push_back(cOutput);
  } else if (dataFormat == FORMAT_NCHW) {
    assitDimInfo.push_back(nOutput);
    assitDimInfo.push_back(cOutput);
    assitDimInfo.push_back(hOutput);
    assitDimInfo.push_back(wOutput);
  }
  return SUCCESS;
}

Status AvgPool1DFusionPass::AvgValueTableGenFp32(vector<int64_t> dimInfo, int64_t kernelSize,
                            int64_t strideSize, vector<int64_t> padding,
                            bool ceilMode, bool countIncludePad,
                            ge::Format dataFormat, ge::DataType inputType,
                            vector<int64_t> &assitDimInfo, float *output) {
  if (output == nullptr) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "output pointer is null!");
      return FAILED;
  }
  if (dimInfo.size() == 0) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "dimInfo is empty!");
      return FAILED;
  }
  if (padding.size() == 0) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "padding is empty!");
      return FAILED;
  }
  int64_t nInput = 1;
  int64_t cInput = 1;
  int64_t hInput = 1;
  int64_t wInInput = 0;

  int64_t kSize = kernelSize;
  int64_t stride = strideSize;
  int64_t padl = padding[0];
  int64_t padr = padding[1];

  // dimInfo must NCHW
  wInInput = dimInfo[3];

  int64_t nOutput = nInput;
  int64_t cOutput = cInput;
  int64_t hOutput = hInput;
  int64_t wOutput = 0;

  if (ceilMode) {
    wOutput = (wInInput + padl + padr - kSize + stride - 1) / stride + 1;
  } else {
    wOutput = ((wInInput + padl + padr) - kSize) / stride + 1;
  }
  if (padl) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    // existing bug in pytorch code
    // padl = 0 and stride is big, but kernel is small, return nan
    if (((wOutput - 1) * stride) >= (wInInput + padl)) {
      wOutput--;
    }
  }

  padr = (wOutput - 1) * stride + kSize - wInInput - padl;

  // set output data
  float dataNum = 0.0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t outOffsetPoint = 0;
  for (int64_t n = 0; n < nOutput; n++) {
    for (int64_t c = 0; c < cOutput; c++) {
      for (int64_t h = 0; h < hOutput; h++) {
        for (int64_t w = 0; w < wOutput; w++) {
          start = stride * w;
          end = stride * w + kSize;
          outOffsetPoint = n * cOutput * hOutput * wOutput +
                           c * hOutput * wOutput + h * wOutput + w;
          if (!countIncludePad) {
            start = max(start, padl);
            end = min(end, wInInput + padl);
          } else {
            end = min(end, wInInput + padl + padr);
          }
          dataNum = end - start;
          float tmp;
          tmp = 1.0 / dataNum;
          output[outOffsetPoint] = tmp;
        }
      }
    }
  }

  if (dataFormat == FORMAT_NHWC) {
    assitDimInfo.push_back(nOutput);
    assitDimInfo.push_back(hOutput);
    assitDimInfo.push_back(wOutput);
    assitDimInfo.push_back(cOutput);
  } else if (dataFormat == FORMAT_NCHW) {
    assitDimInfo.push_back(nOutput);
    assitDimInfo.push_back(cOutput);
    assitDimInfo.push_back(hOutput);
    assitDimInfo.push_back(wOutput);
  }
  return SUCCESS;
}

vector<FusionPattern *> AvgPool1DFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("AvgPool1DFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);
  pattern->AddOpDesc(PATTERN_AVGPOOL1D, {AVGPOOL1D})
      .SetOutput(PATTERN_AVGPOOL1D);
  patterns.push_back(pattern);
  return patterns;
}

Status AvgPool1DFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                   vector<ge::NodePtr> &fusionNodes) {
  std::string fusionOpType = "AvgPool1D";
  // avg_pool_1d node
  ge::NodePtr avgPool1dFussedNode =
      GetNodeFromMapping(PATTERN_AVGPOOL1D, mapping);
  FUSION_PASS_CHECK(avgPool1dFussedNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "avgPool1dFussedNode is null, fusion failed."),
           return PARAM_INVALID);

  // input of avg_pool_1d
  ge::OpDescPtr avgPool1dDesc = avgPool1dFussedNode->GetOpDesc();
  FUSION_PASS_CHECK(avgPool1dDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "avgPool1dFussedNode's OpDesc is null, fusion failed."),
           return PARAM_INVALID);

  string dataFormat;
  int64_t kSize;
  int64_t strides;
  vector<int64_t> pads;
  bool ceilMode;
  bool countIncludePad;

  // get ksize pads strides value ceilMode countIncludePad
  ge::AttrUtils::GetInt(avgPool1dDesc, "ksize", kSize);
  ge::AttrUtils::GetInt(avgPool1dDesc, "strides", strides);
  ge::AttrUtils::GetListInt(avgPool1dDesc, "pads", pads);
  ge::AttrUtils::GetBool(avgPool1dDesc, "ceil_mode", ceilMode);
  ge::AttrUtils::GetBool(avgPool1dDesc, "count_include_pad", countIncludePad);

  // get const input_shape desc, dtype, format, dims
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(avgPool1dFussedNode);

  // gen avgtable matrix
  ge::GeTensorDesc avgPool1dInputShapeTensor =
      avgPool1dFussedNode->GetOpDesc()->GetInputDesc(0);
  ge::DataType inputType = avgPool1dInputShapeTensor.GetDataType();
  FUSION_PASS_CHECK((inputType != DT_FLOAT16 && inputType != DT_FLOAT),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "matrix only support float16 and float32"), return FAILED);
  ge::GeShape avgPool1dShape = avgPool1dInputShapeTensor.GetShape();
  vector<int64_t> avgPool1dDimInfo = avgPool1dShape.GetDims();
  ge::Format inputFormat = avgPool1dInputShapeTensor.GetFormat();
  if (pads.size() != 2) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "pads must list of 2 element.");
    return FAILED;
  }
  if (inputFormat == FORMAT_NCHW) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPool1D inputFormat NCHW.");
  } else if (inputFormat == FORMAT_NHWC) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "AvgPool1D inputFormat NHWC.");
    int64_t origInputShapeH = 0;
    int64_t origInputShapeW = 0;
    int64_t origInputShapeC = 0;
    origInputShapeH = avgPool1dDimInfo[1];
    origInputShapeW = avgPool1dDimInfo[2];
    origInputShapeC = avgPool1dDimInfo[3];
    avgPool1dDimInfo[1] = origInputShapeC;
    avgPool1dDimInfo[2] = origInputShapeH;
    avgPool1dDimInfo[3] = origInputShapeW;
  }
  vector<int64_t> origInputShapeV;
  origInputShapeV.push_back((int64_t)avgPool1dDimInfo[0]);
  origInputShapeV.push_back((int64_t)avgPool1dDimInfo[1]);
  origInputShapeV.push_back((int64_t)avgPool1dDimInfo[2]);
  origInputShapeV.push_back((int64_t)avgPool1dDimInfo[3]);
  // origInputShapeV must NCHW
  ge::GeTensorPtr AvgTableAssitPtr = nullptr;
  int64_t valueTableSize = avgPool1dDimInfo[3];

  FUSION_PASS_CHECK((((avgPool1dDimInfo[1] * avgPool1dDimInfo[2] *
              avgPool1dDimInfo[3]) == 0) ||
            (valueTableSize <= 0)),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "valueTableSize have 0 element"), return FAILED);
  vector<int64_t> avgPool1dAssitDimInfo;
  Status ret;
  if (inputType == DT_FLOAT16) {
    unique_ptr<uint16_t> inputAssit(new (std::nothrow)
                                        uint16_t[valueTableSize]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
             return PARAM_INVALID);
    ret = AvgValueTableGen(origInputShapeV, kSize, strides, pads, ceilMode,
                           countIncludePad, inputFormat, inputType,
                           avgPool1dAssitDimInfo, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);
    ge::GeShape avpPool1dAssitShape(avgPool1dAssitDimInfo);
    // set tensorDesc
    ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NCHW, inputType);
    tensorDesc.SetFormat(inputFormat);
    tensorDesc.SetOriginFormat(inputFormat);
    tensorDesc.SetShape(avpPool1dAssitShape);
    tensorDesc.SetOriginShape(avpPool1dAssitShape);
    FUSION_PASS_MAKE_SHARED(
        (AvgTableAssitPtr = std::make_shared<ge::GeTensor>(
             tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
             valueTableSize * sizeof(uint16_t))),
        AvgTableAssitPtr = nullptr;
        return PARAM_INVALID);
    vector<ge::GeTensorPtr> avgPool1dWeights = {AvgTableAssitPtr};
    ge::OpDescUtils::SetWeights(avgPool1dFussedNode, avgPool1dWeights);

    auto avgPool1dConstInputNodes =
        OpDescUtils::GetConstInputs(avgPool1dFussedNode);
    NodePtr avgPool1dConstInput = avgPool1dConstInputNodes[0];
    avgPool1dConstInput->GetOpDesc()->SetType(CONSTANTOP);
    avgPool1dDesc->SetType("AvgPool1DD");
    return SUCCESS;
  } else if (inputType == DT_FLOAT) {
    unique_ptr<float> inputAssit(new (std::nothrow) float[valueTableSize]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
             return PARAM_INVALID);
    ret = AvgValueTableGenFp32(origInputShapeV, kSize, strides, pads, ceilMode,
                               countIncludePad, inputFormat, inputType,
                               avgPool1dAssitDimInfo, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);
    ge::GeShape avpPool1dAssitShape(avgPool1dAssitDimInfo);
    // set tensorDesc
    ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NCHW, inputType);
    tensorDesc.SetFormat(inputFormat);
    tensorDesc.SetOriginFormat(inputFormat);
    tensorDesc.SetShape(avpPool1dAssitShape);
    tensorDesc.SetOriginShape(avpPool1dAssitShape);
    FUSION_PASS_MAKE_SHARED(
        (AvgTableAssitPtr = std::make_shared<ge::GeTensor>(
             tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
             valueTableSize * sizeof(float))),
        AvgTableAssitPtr = nullptr;
        return PARAM_INVALID);
    vector<ge::GeTensorPtr> avgPool1dWeights = {AvgTableAssitPtr};
    ge::OpDescUtils::SetWeights(avgPool1dFussedNode, avgPool1dWeights);

    auto avgPool1dConstInputNodes =
        OpDescUtils::GetConstInputs(avgPool1dFussedNode);
    NodePtr avgPool1dConstInput = avgPool1dConstInputNodes[0];
    avgPool1dConstInput->GetOpDesc()->SetType(CONSTANTOP);
    avgPool1dDesc->SetType("AvgPool1DD");
    return SUCCESS;
  }
  return SUCCESS;
}
REGISTER_PASS("AvgPool1DFusionPass", BUILT_IN_GRAPH_PASS, AvgPool1DFusionPass);
}  // namespace fe
