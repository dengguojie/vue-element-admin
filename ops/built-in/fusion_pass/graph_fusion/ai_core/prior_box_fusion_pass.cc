/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file prior_box_fusion_pass.cpp
 * \brief
 */
#include "prior_box_fusion_pass.h"

#include <math.h>
#include <iostream>
#include <map>
#include <algorithm>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "common/debug/log.h"

namespace fe {
static const char PATTERN_PRIORBOX[] = "PriorBox";
static const char PRIORBOX[] = "PriorBox";
static const float FLOAT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;

Status BoxValueGenFP16(vector<int64_t> dimInfo, vector<float> data, uint16_t* output) {
  if (output == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT("PriorBoxDV2", "output pointer is null!");
    return FAILED;
  }
  GE_CHECK_POSITIVE_SIZE_RANGE(dimInfo.size());
  if (dimInfo.size() < 4) {
    VECTOR_FUSION_INNER_ERR_REPORT("PriorBoxDV2", "PriorBoxPass output dim size must greater than 3!");
    return FAILED;
  }
  int64_t nInput = dimInfo[0];
  int64_t cInput = dimInfo[1];
  int64_t hInput = dimInfo[2];
  int64_t wInput = dimInfo[3];

  int64_t nOutput = nInput;
  int64_t cOutput = cInput;
  int64_t hOutput = hInput;
  int64_t wOutput = wInput;

  // set output data
  int64_t outOffsetPoint = 0;

  for (int64_t n = 0; n < nOutput; n++) {
    for (int64_t c = 0; c < cOutput; c++) {
      int64_t offset = hOutput * c;
      for (int64_t w = 0; w < wOutput; w++) {
        for (int64_t h = 0; h < hOutput; h++) {
          outOffsetPoint = n * cOutput * hOutput * wOutput + c * hOutput * wOutput + h * wOutput + w;
          fp16_t tmp;
          tmp = data[h + offset];
          output[outOffsetPoint] = tmp.val;
        }
      }
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> PriorBoxPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PriorBoxFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_PRIORBOX, {PRIORBOX}).SetOutput(PATTERN_PRIORBOX);

  patterns.push_back(pattern);

  return patterns;
}

Status PriorBoxPass::ComputeBoxes(int64_t layer_w, int64_t layer_h, int64_t img_w, int64_t img_h, float step_w,
                                  float step_h, vector<float>& min_size, vector<float>& max_size,
                                  vector<float>& variance, vector<float>& aspect_ratios, float offset, bool clip,
                                  int64_t prior_num, vector<float>& output) {
  // Output data's size
  int dim = layer_h * layer_w * prior_num * 4;

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Data size is %d = layer_width[%d] * layer_height[%d] * priornum[%d] * 4 points.", dim,
          layer_w, layer_h, prior_num);

  // Compute prior box
  for (int h = 0; h < layer_h; ++h) {
    for (int w = 0; w < layer_w; ++w) {
      // Using each point in the feature map as a center point,
      // the default offset is 0.5, can be thought to a little offset
      // here map the center point to the orignal
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;

      float box_width = 0.0;
      float box_height = 0.0;
      for (unsigned int s = 0; s < min_size.size(); ++s) {
        int min_value = min_size[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_value;
        // min_size can determine the normalized square box.
        // xmin
        float point_x_min = (center_x - box_width / 2.) / img_w;
        // ymin
        float point_y_min = (center_y - box_height / 2.) / img_h;
        // xmax
        float point_x_max = (center_x + box_width / 2.) / img_w;
        // ymax
        float point_y_max = (center_y + box_height / 2.) / img_h;
        output.push_back(point_x_min);
        output.push_back(point_y_min);
        output.push_back(point_x_max);
        output.push_back(point_y_max);

        if (max_size.size() > 0) {
          int max_value = max_size[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_value * max_value);
          // xmin
          float point_x_min = (center_x - box_width / 2.) / img_w;
          // ymin
          float point_y_min = (center_y - box_height / 2.) / img_h;
          // xmax
          float point_x_max = (center_x + box_width / 2.) / img_w;
          // ymax
          float point_y_max = (center_y + box_height / 2.) / img_h;
          output.push_back(point_x_min);
          output.push_back(point_y_min);
          output.push_back(point_x_max);
          output.push_back(point_y_max);
        }
        // rest of priors
        for (unsigned int r = 0; r < aspect_ratios.size(); ++r) {
          float ar = aspect_ratios[r];
          // by definition, aspect_ratio and min_size codetermine the rectangle box
          box_width = min_value * sqrt(ar);
          box_height = min_value / sqrt(ar);
          // xmin
          float point_x_min = (center_x - box_width / 2.) / img_w;
          // ymin
          float point_y_min = (center_y - box_height / 2.) / img_h;
          // xmax
          float point_x_max = (center_x + box_width / 2.) / img_w;
          // ymax
          float point_y_max = (center_y + box_height / 2.) / img_h;
          output.push_back(point_x_min);
          output.push_back(point_y_min);
          output.push_back(point_x_max);
          output.push_back(point_y_max);
        }
      }
    }
  }
  // clip represants if do any thing about out-of-bounds
  // the default clip is false
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip) {
    for (int d = 0; d < dim; ++d) {
      output[d] = std::min<float>(std::max<float>(output[d], 0.), 1.);
    }
  }

  // the output c dim is 2
  // and the first part is proposal boxe, another part is variance
  if (variance.size() == 1) {
    for (int i = 0; i < dim; ++i)
      output.push_back(variance[0]);
  } else {
    if (variance.size() != 4) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass variance dim size must be 1 or 4!");
      return FAILED;
    }
    int count = 0;
    for (int h = 0; h < layer_h; ++h) {
      for (int w = 0; w < layer_w; ++w) {
        for (int i = 0; i < prior_num; ++i) {
          for (int j = 0; j < 4; ++j) {
            output.push_back(variance[j]);
            ++count;
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status PriorBoxPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into PriorBoxPass");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_PRIORBOX, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of priorbox
  ge::OpDescPtr priorboxDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(priorboxDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // Get inputs
  ge::GeTensorDesc priorboxInputTensor = fusedNode->GetOpDesc()->GetInputDesc(0);

  // Get shape info of input
  ge::GeShape diagInputShape = priorboxInputTensor.GetShape();

  // GESHAPE to vector
  vector<int64_t> dimInfo = diagInputShape.GetDims();
  if (dimInfo.size() == 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dimInfo:%d,%d,%d,%d", dimInfo[0], dimInfo[1], dimInfo[2],
            dimInfo[3]);
  } else if (dimInfo.size() == 5) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dimInfo:%d,%d,%d,%d,%d", dimInfo[0], dimInfo[1], dimInfo[2],
            dimInfo[3], dimInfo[4]);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dim size must be 4 or 5!");
    return FAILED;
  }

  // get image width and height from second input value
  ge::GeTensorDesc imgInputTensor = fusedNode->GetOpDesc()->GetInputDesc(1);
  // get shape information of image
  ge::GeShape imgInputShape = imgInputTensor.GetShape();
  // get dims
  vector<int64_t> imgDimInfo = imgInputShape.GetDims();
  if (imgDimInfo.size() == 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass img dimInfo:%d,%d,%d,%d", imgDimInfo[0], imgDimInfo[1], imgDimInfo[2],
            imgDimInfo[3]);
  } else if (imgDimInfo.size() == 5) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass img dimInfo:%d,%d,%d,%d,%d", imgDimInfo[0], imgDimInfo[1],
            imgDimInfo[2], imgDimInfo[3], imgDimInfo[4]);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass img dim size must be 4 or 5!");
    return FAILED;
  }

  // Get data type
  vector<float> aspect_ratio;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "aspect_ratio", aspect_ratio);
  int64_t ar_size = aspect_ratio.size();
  vector<float> min_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "min_size", min_size);
  int64_t min_size_size = min_size.size();
  vector<float> max_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "max_size", max_size);
  int64_t max_size_size = max_size.size();
  bool flip = true;
  ge::AttrUtils::GetBool(fusedNode->GetOpDesc(), "flip", flip);
  // Get value from prior box
  vector<float> variance;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "variance", variance);
  int64_t img_h = 0;
  ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "img_h", img_h);
  int64_t img_w = 0;
  ge::AttrUtils::GetInt(fusedNode->GetOpDesc(), "img_w", img_w);
  float step_w = 0.0;
  ge::AttrUtils::GetFloat(fusedNode->GetOpDesc(), "step_w", step_w);
  float step_h = 0.0;
  ge::AttrUtils::GetFloat(fusedNode->GetOpDesc(), "step_h", step_h);
  bool clip = false;
  ge::AttrUtils::GetBool(fusedNode->GetOpDesc(), "clip", clip);
  float offset = 0.0;
  ge::AttrUtils::GetFloat(fusedNode->GetOpDesc(), "offset", offset);

  GE_CHECK_POSITIVE_SIZE_RANGE(aspect_ratio.size());
  GE_CHECK_POSITIVE_SIZE_RANGE(min_size.size());

  vector<float> aspectratios_new;
  for (int i = 0; i < ar_size; i++) {
    float ar = aspect_ratio[i];
    bool already_exist = false;
    if (fabsf(ar - 1.0) < 1e-6) {
      already_exist = true;
    } else {
      for (uint16_t j = 0; j < aspectratios_new.size(); j++) {
        if (fabsf(ar - aspectratios_new[j]) < 1e-6) {
          already_exist = true;
          break;
        }
      }
    }
    if (!already_exist) {
      aspectratios_new.push_back(ar);
      if (flip) {
        aspectratios_new.push_back(1.0 / ar);
      }
    }
  }
  int64_t ar_new_size = aspectratios_new.size();

  int64_t priorNum = 0;
  if (ar_size == 1 && (fabsf(aspect_ratio[0] - 1.0) < 1e-6)) {
    priorNum = min_size_size * ar_size + max_size_size;
  } else {
    priorNum = min_size_size + min_size_size * ar_new_size + max_size_size;
  }

  // set layer width and height
  int64_t layer_width = dimInfo[3];
  int64_t layer_height = dimInfo[2];
  if (PatternFusionUtil::IsUnknownShape(layer_width) || PatternFusionUtil::IsUnknownShape(layer_height)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass cannot be applied for unknown shape.");
    return FAILED;
  }
  float step_w_size = 0.0;
  float step_h_size = 0.0;
  // set image width and height
  int64_t img_width = 0;
  int64_t img_height = 0;
  // If img_w and img_h is none in attribute, set width and height from img's dims
  if (img_h == 0 || img_w == 0) {
    // The width and height of input
    img_width = imgDimInfo[3];
    img_height = imgDimInfo[2];
    if (PatternFusionUtil::IsUnknownShape(img_width) || PatternFusionUtil::IsUnknownShape(img_height)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass cannot be applied for unknown shape.");
      return FAILED;
    }
  } else {
    img_width = img_w;
    img_height = img_h;
  }
  // If step_w and step_h is none in attribute, set width and height by layer
  if (step_w == 0 || step_h == 0) {
    step_w_size = static_cast<float>(img_width) / layer_width;  // scaling
    step_h_size = static_cast<float>(img_height) / layer_height;
  } else {
    step_w_size = step_w;
    step_h_size = step_h;
  }

  vector<float> outputData;
  // Compute prior box result
  Status ret = ComputeBoxes(layer_width, layer_height, img_width, img_height, step_w_size, step_h_size, min_size,
                            max_size, variance, aspectratios_new, offset, clip, priorNum, outputData);

  // The prior box input shape, same with output desc
  ge::GeTensorDesc boxOutTensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  // prior box input tensor, same as output tensor
  ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
  boxOutTensorDesc.SetOriginFormat(outputDesc.GetOriginFormat());
  boxOutTensorDesc.SetFormat(outputDesc.GetFormat());
  boxOutTensorDesc.SetOriginDataType(outputDesc.GetOriginDataType());
  boxOutTensorDesc.SetDataType(outputDesc.GetDataType());
  boxOutTensorDesc.SetOriginShape(outputDesc.GetOriginShape());
  boxOutTensorDesc.SetShape(outputDesc.GetShape());
  // output dims
  vector<int64_t> outputDims = boxOutTensorDesc.GetShape().GetDims();

  if (outputDims.size() < 4) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass output dim size must greater than 3!");
    return FAILED;
  }
  for (size_t i = 0; i <= 3; i++) {
    auto dim = outputDims[i];
    if (PatternFusionUtil::IsUnknownShape(dim)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PriorBoxPass cannot be applied for unknown shape.");
      return FAILED;
    }
  }
  int64_t dimNums = outputDims[0] * outputDims[1] * outputDims[2] * outputDims[3];

  ge::GeTensorPtr assitPtr = nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Input type is FP16.");
  unique_ptr<uint16_t[]> outputAssit(new (std::nothrow) uint16_t[dimNums]());
  FUSION_PASS_CHECK(outputAssit.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outputAssit is NULL"),
                    return PARAM_INVALID);
  ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(outputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);
  // vector to tensor for const
  ret = BoxValueGenFP16(outputDims, outputData, outputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, 
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Generate data by FP16 failed."), return ret);
  // set output type to fp16
  boxOutTensorDesc.SetDataType(ge::DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(boxOutTensorDesc, reinterpret_cast<uint8_t*>(outputAssit.get()),
                                                 dimNums * sizeof(uint16_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(fusedNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(fusedNode);
  NodePtr constInput0 = constInputNodes[0];
  constInput0->GetOpDesc()->SetType("Const");
  // set type, use priorboxdv2 op
  priorboxDesc->SetType("PriorBoxDV2");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxFusionPass pass handle success!!!!");

  return SUCCESS;
}
REGISTER_PASS("PriorBoxPass", BUILT_IN_GRAPH_PASS, PriorBoxPass);
}  // namespace fe
